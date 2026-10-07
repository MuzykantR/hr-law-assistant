"""Access to Gemini (chat + embeddings) through an OpenAI-compatible API.

Two providers, picked by which key is set in .env:
- GEMINI_API_KEY      -> Google AI Studio directly (preferred: no middleman, free tier);
- OPENROUTER_API_KEY  -> the same Gemini models through OpenRouter (model ids get a "google/" prefix).

Both speak the OpenAI protocol, so one client class covers them. The rest of the code depends only
on embed() and chat(), so switching the provider (GigaChat, a local model) means changing this file only.
"""

import os
import re
import time
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from openai import APITimeoutError, InternalServerError, OpenAI, RateLimitError

load_dotenv()

PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", ""),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "google/"),
}


def _pick_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER")
    if explicit:
        return explicit
    return "gemini" if os.getenv("GEMINI_API_KEY") else "openrouter"


PROVIDER = _pick_provider()
_KEY_VAR, BASE_URL, _PREFIX = PROVIDERS[PROVIDER]
LLM_MODEL = os.getenv("LLM_MODEL", _PREFIX + "gemini-3.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", _PREFIX + "gemini-embedding-2")


@lru_cache(maxsize=1)
def client() -> OpenAI:
    key = os.getenv(_KEY_VAR)
    if not key:
        raise RuntimeError(f"{_KEY_VAR} is not set — copy .env.example to .env and fill it in")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=60, max_retries=1)


def embed(texts: list[str], model: str = EMBED_MODEL, batch_size: int = 64) -> np.ndarray:
    """Return L2-normalized embeddings, shape (len(texts), dim).

    Normalizing once here means cosine similarity later is just a dot product.
    """
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = _with_rate_limit_retry(lambda: client().embeddings.create(model=model, input=texts[i : i + batch_size]))
        # Google's endpoint returns items in input order but without `index`; OpenRouter fills it in.
        data = resp.data if any(d.index is None for d in resp.data) else sorted(resp.data, key=lambda d: d.index)
        vectors.extend(d.embedding for d in data)
    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


FALLBACK_MODELS = [_PREFIX + m for m in ("gemini-3.1-flash-lite", "gemini-3.6-flash")]


def chat(messages: list[dict], model: str = LLM_MODEL, temperature: float = 0.1, max_tokens: int = 2000) -> str:
    """Generate an answer. If the model is overloaded (HTTP 503 "high demand", common for new Gemini
    releases), too slow (timeout) or its daily free-tier quota is spent (HTTP 429), fall back to the next
    model — quotas are counted per model — instead of failing the user's request.

    reasoning_effort="low": Gemini 3.x models "think" before answering by default, which on the free tier
    takes 20–60 s. Answering from a handful of retrieved fragments does not need long reasoning."""
    last_error: Exception | None = None
    for candidate in [model, *[m for m in FALLBACK_MODELS if m != model]]:
        try:
            # Without an explicit max_tokens OpenRouter reserves the model's full output budget (~65k tokens)
            # and rejects the request on a small balance; answers here are a few paragraphs at most.
            resp = _with_rate_limit_retry(
                lambda: client().chat.completions.create(
                    model=candidate,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort="low",
                )
            )
            return (resp.choices[0].message.content or "").strip()
        except (APITimeoutError, InternalServerError, RateLimitError) as e:
            last_error = e
    raise last_error


def _with_rate_limit_retry(call, attempts: int = 5, max_wait_s: float = 90):
    """The Gemini free tier allows ~100 embedding requests per minute. On HTTP 429 the API says how long
    to wait ("Please retry in 40.3s"); for such short per-minute limits we sleep and retry.
    A spent daily quota says "retry in 8h57m…" — waiting is pointless, so the error is re-raised
    and chat() moves on to a fallback model."""
    for attempt in range(attempts):
        try:
            return call()
        except RateLimitError as e:
            m = re.search(r"retry in ([\d.]+)s", str(e))
            if attempt == attempts - 1 or not m or float(m.group(1)) > max_wait_s:
                raise
            time.sleep(float(m.group(1)) + 1)
