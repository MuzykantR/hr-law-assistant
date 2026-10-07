"""Three retrievers with the same interface: lexical (BM25), dense (embeddings) and hybrid (RRF).

- BM25 matches words. Strong on exact legal terms ("сверхурочная", "простой"), blind to synonyms:
  "сколько отдыхать в году" shares no words with "ежегодный оплачиваемый отпуск".
- Dense search compares meaning vectors, so it handles paraphrases, but it can miss rare exact terms.
- Hybrid combines both rankings with Reciprocal Rank Fusion: score(d) = sum over lists of 1/(k + rank).
  RRF uses only ranks, so it does not need BM25 scores and cosine similarities to be on one scale.
"""

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from . import llm
from .corpus import Chunk
from .text import tokenize

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return (chunk_id, score) pairs, best first."""
        ...


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], model: str = llm.EMBED_MODEL):
        self.model = model
        self.matrix = load_or_build_embeddings(chunks, model)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = llm.embed([query], self.model)[0]
        sims = self.matrix @ q  # vectors are normalized, so dot product == cosine similarity
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. k=60 is the constant from the original paper (Cormack et al., 2009):
    it damps the gap between rank 1 and rank 2 so that one list cannot dominate.
    Optional weights let a weaker retriever vote with less force."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, retrievers: list[Retriever], weights: list[float] | None = None, depth: int = 50, rrf_k: int = 60):
        self.retrievers = retrievers
        self.weights = weights
        self.depth = depth  # how many candidates each retriever contributes to the fusion
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        rankings = [[i for i, _ in r.search(query, self.depth)] for r in self.retrievers]
        return rrf(rankings, self.rrf_k, self.weights)[:k]


def load_or_build_embeddings(chunks: list[Chunk], model: str) -> np.ndarray:
    """Embed all chunks once and cache to disk; the cache key covers the model and the exact texts.

    The provider prefix is dropped from the key ("google/gemini-embedding-2" == "gemini-embedding-2"):
    OpenRouter forwards to the same Google model, so vectors from either provider are interchangeable.
    """
    texts = [c.text for c in chunks]
    canonical = model.split("/")[-1]
    digest = hashlib.sha256(("\x00".join([canonical, *texts])).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"emb_{canonical}_{digest}.npy"
    if path.exists():
        return np.load(path)
    matrix = llm.embed(texts, model)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix
