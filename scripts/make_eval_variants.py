"""Generate noisy variants of the paraphrase questions for the evaluation set (reproducible, fixed seed).

    layout — the first 12 paraphrase questions retyped on the English keyboard layout;
    typo   — every paraphrase question with realistic typos: a confused vowel (о/а, е/и, е/я),
             a dropped, doubled or swapped letter in roughly every third word of 5+ letters.

Rewrites the layout/typo lines of data/eval_questions.jsonl and keeps the hand-written ones.

Usage:
    uv run python scripts/make_eval_variants.py
"""

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.layout import _JCUKEN, _QWERTY  # noqa: E402

PATH = ROOT / "data" / "eval_questions.jsonl"
TO_QWERTY = str.maketrans(_JCUKEN, _QWERTY)
VOWEL_SWAPS = {"о": "а", "а": "о", "е": "и", "и": "е", "я": "е"}


def add_typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)  # never touch the first letter: people rarely mistype it
    vowels = [j for j in range(1, len(word)) if word[j] in VOWEL_SWAPS]
    op = rng.choice(["vowel", "vowel", "drop", "double", "swap"])
    if op == "vowel" and vowels:
        j = rng.choice(vowels)
        return word[:j] + VOWEL_SWAPS[word[j]] + word[j + 1 :]
    if op == "drop":
        return word[:i] + word[i + 1 :]
    if op == "double":
        return word[: i + 1] + word[i] + word[i + 1 :]
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]  # swap neighbours


def with_typos(question: str, rng: random.Random) -> str:
    words = re.findall(r"[А-Яа-яЁё]{5,}", question)
    chosen = {w for w in words if rng.random() < 0.35} or {rng.choice(words)}
    return re.sub(r"[А-Яа-яЁё]{5,}", lambda m: add_typo(m.group(), rng) if m.group() in chosen else m.group(), question)


def main() -> None:
    rows = [json.loads(line) for line in PATH.open(encoding="utf-8")]
    base = [r for r in rows if r["type"] not in ("layout", "typo")]
    paraphrases = [r for r in base if r["type"] == "paraphrase"]

    rng = random.Random(42)
    layout = [{**r, "question": r["question"].translate(TO_QWERTY), "type": "layout"} for r in paraphrases[:12]]
    typo = [{**r, "question": with_typos(r["question"], rng), "type": "typo"} for r in paraphrases]

    with PATH.open("w", encoding="utf-8") as f:
        for r in [*base, *layout, *typo]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in typo[:8]:
        print(r["question"])
    print(f"{len(base)} hand-written + {len(layout)} layout + {len(typo)} typo questions")


if __name__ == "__main__":
    main()
