"""Compare BM25, dense and hybrid retrieval on questions with a known correct article.

Two question types, because they favour different methods:
    paraphrase — how an employee actually asks ("можно ли разбить отпуск на куски?")
    terms      — short queries in the law's own words ("оплата сверхурочной работы")

A hit means a chunk of a correct article is among the top-k results.
    Recall@k — share of questions where the right article is in the top k.
    MRR@10   — mean of 1/rank of the first correct article (0 if not in the top 10);
               rewards putting the right article first, not just somewhere in the list.

Usage:
    uv run python scripts/evaluate.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.assistant import build_retriever  # noqa: E402
from rag.corpus import load_chunks  # noqa: E402

MODES = ["bm25", "dense", "hybrid"]


def first_hit_rank(articles: list[str], expected: set[str]) -> int | None:
    for rank, article in enumerate(articles, 1):
        if article in expected:
            return rank
    return None


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "recall@1": sum(r == 1 for r in ranks) / n,
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr@10": sum(1 / r for r in ranks if r) / n,
    }


def main() -> None:
    chunks = load_chunks()
    questions = [json.loads(line) for line in (ROOT / "data" / "eval_questions.jsonl").open(encoding="utf-8")]
    missing = {a for q in questions for a in q["articles"]} - {c.article for c in chunks}
    if missing:
        raise SystemExit(f"Eval set references articles absent from the corpus: {sorted(missing)}")

    types = sorted({q["type"] for q in questions}) + ["all"]
    results: dict[str, dict] = {}
    misses: dict[str, list[str]] = {}
    for mode in MODES:
        retriever = build_retriever(chunks, mode)
        ranks, latencies = [], []
        for q in questions:
            t = time.perf_counter()
            hits = retriever.search(q["question"], k=10)
            latencies.append(time.perf_counter() - t)
            rank = first_hit_rank([chunks[i].article for i, _ in hits], set(q["articles"]))
            ranks.append(rank)
            if rank is None or rank > 5:
                misses.setdefault(mode, []).append(f"[{q['type']}] {q['question']}")
        results[mode] = {
            t: metrics([r for r, q in zip(ranks, questions) if t in ("all", q["type"])]) for t in types
        }
        results[mode]["latency_ms"] = 1000 * sorted(latencies)[len(latencies) // 2]

    lines = []
    for t in types:
        n = results[MODES[0]][t]["n"]
        lines += [f"**{t}** ({n} вопросов)", "", "| Поиск | Recall@1 | Recall@5 | MRR@10 |", "|---|---|---|---|"]
        for mode in MODES:
            m = results[mode][t]
            lines.append(f"| {mode} | {m['recall@1']:.2f} | {m['recall@5']:.2f} | {m['mrr@10']:.2f} |")
        lines.append("")
    lines.append("Медианная задержка поиска: " + ", ".join(f"{m} — {results[m]['latency_ms']:.0f} мс" for m in MODES))
    report = "\n".join(lines)

    print(f"{len(questions)} вопросов, {len(chunks)} фрагментов\n")
    print(report)
    for mode, qs in misses.items():
        print(f"\nПромахи top-5 ({mode}):")
        for q in qs:
            print(f"  - {q}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "retrieval_eval.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "retrieval_eval.md").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
