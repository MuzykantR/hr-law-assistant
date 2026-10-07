"""Calibrate the refusal gate: the minimal cosine similarity of the best chunk for a question to be answered.

Questions about the code (data/eval_questions.jsonl) must pass, off-topic ones (data/offtopic_questions.jsonl)
should be refused without calling the LLM. Both sets are split in half by position:
    calibration — the threshold is chosen here: in the middle of the gap between the weakest in-domain
                  question and the strongest off-topic one, so both sides get the same safety margin;
    holdout     — the threshold is checked on questions it has not seen.

Usage:
    uv run python scripts/calibrate_gate.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.assistant import build_layout_fixer  # noqa: E402
from rag.corpus import load_chunks  # noqa: E402
from rag.retrieval import DenseRetriever  # noqa: E402


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / name).open(encoding="utf-8")]


def main() -> None:
    chunks = load_chunks()
    retriever = DenseRetriever(chunks)
    fixer = build_layout_fixer(chunks)

    def top_score(question: str) -> float:
        return retriever.search(fixer.fix(question), k=1)[0][1]

    in_domain = [(q["question"], top_score(q["question"])) for q in load("eval_questions.jsonl")]
    off_topic = [(q["question"], q["kind"], top_score(q["question"])) for q in load("offtopic_questions.jsonl")]

    calib_in, hold_in = in_domain[0::2], in_domain[1::2]
    calib_off, hold_off = off_topic[0::2], off_topic[1::2]

    weakest_in, strongest_off = min(s for _, s in calib_in), max(s for *_, s in calib_off)
    threshold = round((weakest_in + strongest_off) / 2, 3)

    def report(name: str, ins: list, offs: list) -> list[str]:
        false_refusals = [q for q, s in ins if s < threshold]
        refused = [(q, k) for q, k, s in offs if s < threshold]
        far = [k for _, k, _ in offs if k == "far"]
        near = [k for _, k, _ in offs if k == "near"]
        return [
            f"| {name} | {len(ins) - len(false_refusals)}/{len(ins)} | "
            f"{sum(k == 'far' for _, k in refused)}/{len(far)} | {sum(k == 'near' for _, k in refused)}/{len(near)} |"
        ]

    lines = [
        f"Порог: **{threshold}** — середина зазора на калибровочной половине между самым слабым вопросом "
        f"по кодексу ({weakest_in:.3f}) и самым сильным вопросом не по теме ({strongest_off:.3f})",
        "",
        "| Выборка | Вопросы по кодексу: ответил | Совсем не по теме: отказ | Рядом с темой: отказ |",
        "|---|---|---|---|",
        *report("калибровочная", calib_in, calib_off),
        *report("отложенная", hold_in, hold_off),
        "",
        "Близость лучшего фрагмента:",
        "",
        f"- вопросы по кодексу: от {min(s for _, s in in_domain):.3f} до {max(s for _, s in in_domain):.3f}",
        f"- не по теме: от {min(s for *_, s in off_topic):.3f} до {max(s for *_, s in off_topic):.3f}",
        "",
        "Вопросы не по теме, которые проходят порог (на них откажет уже промпт модели):",
        "",
        *[f"- {q} ({s:.3f})" for q, _, s in sorted(off_topic, key=lambda x: -x[2]) if s >= threshold],
        "",
        "Самые слабые вопросы по кодексу:",
        "",
        *[f"- {q} ({s:.3f})" for q, s in sorted(in_domain, key=lambda x: x[1])[:5]],
    ]
    report_text = "\n".join(lines)
    print(report_text)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "gate_calibration.md").write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
