"""Compare BM25, dense and hybrid retrieval on questions with a known correct article.

Question types (data/eval_questions.jsonl), each stresses a different part of the pipeline:
    paraphrase — how an employee actually asks ("можно ли разбить отпуск на куски?")
    terms      — short queries in the law's own words ("оплата сверхурочной работы")
    layout     — paraphrase questions typed on the English layout ("vj;yj kb hfp,bnm jngecr...")
    typo       — paraphrase questions with typos ("мжоно ли разбить отупск...")
Noisy types are reported twice: through query normalization (layout + typo fixing, as in production)
and without it, which shows what the normalization gives.

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

from rag.assistant import Assistant, build_layout_fixer, build_retriever, build_spelling_fixer  # noqa: E402
from rag.corpus import load_chunks  # noqa: E402

MODES = ["bm25", "dense", "hybrid"]
NOISY = ["layout", "typo"]


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

    normalizer = Assistant(
        chunks, retriever=None, layout_fixer=build_layout_fixer(chunks), spelling_fixer=build_spelling_fixer(chunks)
    )
    normalized = {q["question"]: normalizer.normalize(q["question"]) for q in questions}
    clean_changed = [
        q["question"]
        for q in questions
        if q["type"] in ("paraphrase", "terms") and normalized[q["question"]] != q["question"]
    ]

    types = sorted({q["type"] for q in questions}, key=["paraphrase", "terms", *NOISY].index)
    groups = [*types, "all", *[f"{t} без исправления" for t in NOISY]]
    results: dict[str, dict] = {}
    misses: dict[str, list[str]] = {}
    for mode in MODES:
        retriever = build_retriever(chunks, mode)
        ranks: dict[str, list] = {g: [] for g in groups}
        latencies = []
        for q in questions:
            expected = set(q["articles"])
            t = time.perf_counter()
            hits = retriever.search(normalized[q["question"]], k=10)
            latencies.append(time.perf_counter() - t)
            rank = first_hit_rank([chunks[i].article for i, _ in hits], expected)
            ranks[q["type"]].append(rank)
            ranks["all"].append(rank)
            if q["type"] in NOISY:
                raw_hits = retriever.search(q["question"], k=10)
                raw_rank = first_hit_rank([chunks[i].article for i, _ in raw_hits], expected)
                ranks[f"{q['type']} без исправления"].append(raw_rank)
            if rank is None or rank > 5:
                misses.setdefault(mode, []).append(f"[{q['type']}] {q['question']}")
        results[mode] = {g: metrics(r) for g, r in ranks.items()}
        results[mode]["latency_ms"] = 1000 * sorted(latencies)[len(latencies) // 2]

    lines = []
    for g in groups:
        n = results[MODES[0]][g]["n"]
        lines += [f"**{g}** ({n} вопросов)", "", "| Поиск | Recall@1 | Recall@5 | MRR@10 |", "|---|---|---|---|"]
        for mode in MODES:
            m = results[mode][g]
            lines.append(f"| {mode} | {m['recall@1']:.2f} | {m['recall@5']:.2f} | {m['mrr@10']:.2f} |")
        lines.append("")
    lines.append("Медианная задержка поиска (эмбеддинги вопросов, уже бывших в кеше, не запрашиваются по сети): " + ", ".join(f"{m} — {results[m]['latency_ms']:.0f} мс" for m in MODES))
    lines.append(f"Чистых вопросов, изменённых исправлением запроса: {len(clean_changed)}")
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
