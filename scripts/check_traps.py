"""Ask questions with a false premise ("какая компенсация положена за работу в полнолуние?") and save the answers.

These questions are on-topic, so the score gate lets them through; the risk is that the LLM plays along
and invents a rule. The answers are saved for manual review: an answer is correct if it says that the code
does not provide for the thing in the question, and wrong if it describes it as existing.

Usage:
    uv run python scripts/check_traps.py --out results/traps_after.md
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.assistant import build_assistant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/traps.md")
    args = parser.parse_args()

    assistant = build_assistant()
    traps = [json.loads(line) for line in (ROOT / "data" / "trap_questions.jsonl").open(encoding="utf-8")]
    lines = []
    for t in traps:
        answer = assistant.ask(t["question"])
        lines += [f"### {t['question']}", f"*Ложная предпосылка:* {t['false_premise']}", "", answer.text, ""]
        print(f"--- {t['question']}\n{answer.text}\n")
    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
