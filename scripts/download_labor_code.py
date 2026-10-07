"""Download the Labour Code of the Russian Federation and save it as structured JSONL.

Official legal texts are not subject to copyright (Civil Code of the RF, art. 1259 p. 6),
so the resulting file is committed to the repo and this script only needs to run once.

Output: data/labor_code.jsonl — one line per article:
    {"article": "115", "title": "...", "chapter": "Глава 19. Отпуска", "parts": ["...", "..."]}

Usage:
    uv run python scripts/download_labor_code.py
"""

import argparse
import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE = "https://www.consultant.ru"
TOC_URL = f"{BASE}/document/cons_doc_LAW_34683/"
HEADERS = {"User-Agent": "Mozilla/5.0 (labor-code-rag; educational project)"}

ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)?)\.\s*(.*)$")
CHAPTER_RE = re.compile(r"^Глава\s+\d+(?:\.\d+)?\.")


def parse_toc(html: str) -> list[dict]:
    """Return articles in order, each with the chapter it belongs to."""
    soup = BeautifulSoup(html, "html.parser")
    items, chapter = [], ""
    for a in soup.select('a[href^="/document/cons_doc_LAW_34683/"]'):
        text = " ".join(a.get_text().split())
        if CHAPTER_RE.match(text):
            chapter = text
        elif m := ARTICLE_RE.match(text):
            items.append({"article": m.group(1), "title": m.group(2), "chapter": chapter, "url": BASE + a["href"]})
    # The TOC can repeat links (e.g. in navigation blocks); keep the first occurrence.
    seen, unique = set(), []
    for it in items:
        if it["article"] not in seen:
            seen.add(it["article"])
            unique.append(it)
    return unique


def parse_article(html: str) -> list[str]:
    """Extract the article's parts (paragraphs), skipping editorial notes and ads."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.document-page__content")
    if content is None:
        return []
    # Inserted blocks ("Готовые решения") and amendment notes ("в ред. закона ...") are not law text.
    for junk in content.select("div.doc-insert, div.doc-edit, h1"):
        junk.decompose()
    parts = []
    for p in content.find_all("p"):
        text = " ".join(p.get_text().split())
        if text and p.find("a", id=re.compile(r"^dst")):
            parts.append(text)
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/labor_code.jsonl")
    parser.add_argument("--delay", type=float, default=0.4, help="pause between requests, seconds")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        toc = parse_toc(client.get(TOC_URL).text)
        print(f"TOC: {len(toc)} articles")
        written = 0
        with out.open("w", encoding="utf-8") as f:
            for i, item in enumerate(toc, 1):
                for attempt in range(3):
                    try:
                        parts = parse_article(client.get(item["url"]).text)
                        break
                    except httpx.HTTPError:
                        time.sleep(2 * (attempt + 1))
                else:
                    print(f"  failed: Статья {item['article']}")
                    continue
                # Repealed articles contain only "Утратила силу" — keep them out of the index.
                if not parts or (len(parts) == 1 and parts[0].lower().startswith("утратил")):
                    continue
                record = {k: item[k] for k in ("article", "title", "chapter")} | {"parts": parts}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if i % 50 == 0:
                    print(f"  {i}/{len(toc)}")
                time.sleep(args.delay)
    print(f"Saved {written} articles to {out}")


if __name__ == "__main__":
    main()
