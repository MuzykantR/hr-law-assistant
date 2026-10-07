"""Retrieval-augmented answering: find relevant parts of the code, then let the LLM answer from them only."""

from dataclasses import dataclass

from . import llm
from .corpus import Chunk, load_chunks
from .layout import LayoutFixer, vocabulary_from_texts
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever

NOT_FOUND = "В Трудовом кодексе я не нашёл ответа на этот вопрос."

SYSTEM_PROMPT = f"""Ты — HR-ассистент, который отвечает сотрудникам на вопросы по Трудовому кодексу РФ.

Правила:
1. Отвечай ТОЛЬКО на основе фрагментов кодекса, приведённых в сообщении. Не используй другие знания.
2. После каждого утверждения ставь ссылку на источник в квадратных скобках точно в том виде,
   в котором она указана у фрагмента, например [ст. 115, ч. 1].
3. Если во фрагментах нет ответа, ответь ровно: «{NOT_FOUND}» — и ничего не придумывай.
4. Пиши коротко и по-человечески: сначала прямой ответ, потом важные условия и исключения."""


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    query: str  # the question after layout fixing; differs from the input if it was typed in the wrong layout


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.citation}]\n{c.text}" for c in chunks)


class Assistant:
    def __init__(self, chunks: list[Chunk], retriever: Retriever, layout_fixer: LayoutFixer | None = None, top_k: int = 6):
        self.chunks = chunks
        self.retriever = retriever
        self.layout_fixer = layout_fixer
        self.top_k = top_k

    def normalize(self, question: str) -> str:
        return self.layout_fixer.fix(question) if self.layout_fixer else question

    def retrieve(self, question: str) -> list[Chunk]:
        return [self.chunks[i] for i, _ in self.retriever.search(question, self.top_k)]

    def ask(self, question: str) -> Answer:
        query = self.normalize(question)
        sources = self.retrieve(query)
        if not sources:
            return Answer(NOT_FOUND, [], query)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Фрагменты кодекса:\n\n{build_context(sources)}\n\nВопрос: {query}"},
        ]
        text = llm.chat(messages)
        # Show sources only if the model actually answered from them.
        return Answer(text, [] if text.startswith(NOT_FOUND[:20]) else sources, query)


def build_retriever(chunks: list[Chunk], mode: str = "hybrid") -> Retriever:
    if mode == "bm25":
        return BM25Retriever(chunks)
    if mode == "dense":
        return DenseRetriever(chunks)
    if mode == "hybrid":
        return HybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks)])
    raise ValueError(f"unknown retrieval mode: {mode}")


def build_layout_fixer(chunks: list[Chunk]) -> LayoutFixer:
    return LayoutFixer(vocabulary_from_texts([c.text for c in chunks]))


def build_assistant(mode: str = "dense") -> Assistant:
    chunks = load_chunks()
    return Assistant(chunks, build_retriever(chunks, mode), build_layout_fixer(chunks))
