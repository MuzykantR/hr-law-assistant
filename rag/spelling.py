"""Fixing typos in a query: "сколько дней отпуско мне положеон" -> "сколько дней отпуска мне положено".

Two dictionaries with different jobs:
- the Russian language dictionary (OpenCorpora via pymorphy3) answers "is this a real word?" —
  only words that do not exist in Russian are touched, so "положено" or "зарплату" stay as they are
  even though the Labour Code never uses these exact forms;
- the vocabulary of the Labour Code is the preferred source of corrections: for "отпуско" both "отпуска"
  and "отпуск" are real words, the code's own word forms win.

Candidates are words at edit distance 1 (a typo usually is one wrong, missing, extra or swapped letter);
for words of 8+ letters, corpus words at distance 2 are also considered.

Same idea as the earlier hackathon prototype (experiments EXP-001/003: fuzzy matching against a term
dictionary), with a real language dictionary added to avoid "fixing" correct words.
"""

import re
from collections import defaultdict
from functools import lru_cache

import pymorphy3
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .text import STOPWORDS, tokenize

_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)
_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя"


@lru_cache(maxsize=1)
def _morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


def _norm(word: str) -> str:
    return word.lower().replace("ё", "е")


def _edits1(w: str) -> set[str]:
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    deletes = {a + b[1:] for a, b in splits if b}
    swaps = {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    replaces = {a + c + b[1:] for a, b in splits if b for c in _ALPHABET}
    inserts = {a + c + b for a, b in splits for c in _ALPHABET}
    return deletes | swaps | replaces | inserts


class SpellingFixer:
    def __init__(self, texts: list[str]):
        words = {_norm(w) for text in texts for w in _WORD_RE.findall(text)}
        self.words = words
        self.stems = {s for w in words for s in tokenize(w)}
        self.by_length: dict[int, list[str]] = defaultdict(list)
        for w in words:
            self.by_length[len(w)].append(w)

    def is_known(self, w: str) -> bool:
        return w in self.words or w in STOPWORDS or _morph().word_is_known(w)

    def fix(self, query: str) -> str:
        return _WORD_RE.sub(lambda m: self._fix_word(m.group()), query)

    def _fix_word(self, word: str) -> str:
        w = _norm(word)
        if len(w) < 4 or self.is_known(w):
            return word
        fixed = self._best_candidate(w)
        if fixed is None:
            return word
        return fixed.capitalize() if word[0].isupper() else fixed

    def _best_candidate(self, w: str) -> str | None:
        edits = _edits1(w)
        in_corpus = sorted(e for e in edits if e in self.words)
        if in_corpus:
            return in_corpus[0]
        if len(w) >= 8:
            candidates = [c for n in range(len(w) - 2, len(w) + 3) for c in self.by_length[n]]
            best = process.extractOne(w, candidates, scorer=Levenshtein.distance, score_cutoff=2)
            if best is not None:
                return best[0]
        real_words = sorted(e for e in edits if _morph().word_is_known(e))
        if not real_words:
            return None
        # Among real words prefer ones whose stem occurs in the code: closer to the domain.
        in_domain = [e for e in real_words if any(s in self.stems for s in tokenize(e))]
        return (in_domain or real_words)[0]
