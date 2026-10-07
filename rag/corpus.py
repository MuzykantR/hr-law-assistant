"""Loading the Labour Code and splitting it into citable chunks.

Why structure-aware chunking: a legal answer is only useful if it points to an exact place
("ст. 115, ч. 1"). Fixed-size windows would cut articles in the middle and lose that address,
so we split along the document's own structure: article -> parts. Each chunk keeps the article
number and the range of parts it covers.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Paragraphs like "1) ликвидации организации..." or "а) ..." are list items inside a part,
# not separate parts, so they are glued to the preceding paragraph.
LIST_ITEM_RE = re.compile(r"^(\d+(\.\d+)?|[а-я])\)\s")

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "labor_code.jsonl"


@dataclass(frozen=True)
class Chunk:
    id: int
    article: str
    title: str
    chapter: str
    part_start: int  # 1-based, inclusive
    part_end: int
    parts: tuple[str, ...]

    @property
    def citation(self) -> str:
        if self.part_start == self.part_end:
            return f"ст. {self.article}, ч. {self.part_start}"
        return f"ст. {self.article}, ч. {self.part_start}–{self.part_end}"

    @property
    def text(self) -> str:
        """Text used for indexing and as LLM context: chapter + header + numbered parts.

        The chapter matters for articles with generic titles: "Статья 312.1. Общие положения"
        says nothing by itself, "Глава 49.1. Особенности регулирования труда дистанционных работников" does.
        """
        body = "\n".join(f"(ч. {self.part_start + i}) {p}" for i, p in enumerate(self.parts))
        return f"{self.chapter}\nСтатья {self.article}. {self.title}\n{body}"


def group_parts(paragraphs: list[str]) -> list[str]:
    """Merge list-item paragraphs into the part they belong to."""
    parts: list[str] = []
    for p in paragraphs:
        if parts and LIST_ITEM_RE.match(p):
            parts[-1] += "\n" + p
        else:
            parts.append(p)
    return parts


def chunk_article(record: dict, max_chars: int = 1500) -> list[dict]:
    """Split one article into chunks of consecutive parts, each at most ~max_chars.

    Short articles (the majority) become a single chunk. A part longer than max_chars
    is never cut — it becomes its own chunk, because cutting would break the citation.
    """
    parts = group_parts(record["parts"])
    chunks, current, start = [], [], 1
    for i, part in enumerate(parts, 1):
        if current and sum(len(p) for p in current) + len(part) > max_chars:
            chunks.append({"part_start": start, "part_end": i - 1, "parts": current})
            current, start = [], i
        current.append(part)
    if current:
        chunks.append({"part_start": start, "part_end": len(parts), "parts": current})
    return chunks


def load_chunks(path: Path = DEFAULT_CORPUS, max_chars: int = 1500) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for c in chunk_article(record, max_chars):
                chunks.append(
                    Chunk(
                        id=len(chunks),
                        article=record["article"],
                        title=record["title"],
                        chapter=record["chapter"],
                        part_start=c["part_start"],
                        part_end=c["part_end"],
                        parts=tuple(c["parts"]),
                    )
                )
    return chunks
