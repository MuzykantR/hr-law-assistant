"""Fixing queries typed in the wrong keyboard layout: "crjkmrj lytq jngecrf" -> "сколько дней отпуска".

A Russian word typed with the English layout is Latin gibberish: BM25 finds nothing, and embeddings
see a meaningless string. The fix works per token, so a query that mixes both layouts
("сколько дней jngecrf") is repaired without touching the correct words.

Ported from the earlier hackathon prototype (experiment EXP-004, token-level layout inversion).
Same rules, plus one data-driven check that prototype did not have: a token is flipped if its
Cyrillic version is a word from the corpus vocabulary, which catches most real cases without guessing.
"""

import re

from .text import STOPWORDS, tokenize

_QWERTY = "qwertyuiop[]asdfghjkl;'zxcvbnm,./" 'QWERTYUIOP{}ASDFGHJKL:"ZXCVBNM<>?' "`~@#$^&"
_JCUKEN = "йцукенгшщзхъфывапролджэячсмитьбю." "ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ," 'ёЁ"№;:?'
QWERTY_TO_JCUKEN = str.maketrans(_QWERTY, _JCUKEN)

CYRILLIC_VOWELS = set("аеёиоуыэюя")
# Keys that are punctuation in the English layout but letters in Russian (х, ъ, ж, э, б, ю, ё).
LAYOUT_PUNCTUATION = set(",.;'[]`~<>:\"{}")
# Sentence punctuation around a word; kept as is.
BOUNDARY_CHARS = set("?!:()«»\"“”„/\\|")
# Latin words that are legitimately Latin in a question and must not be flipped.
KEEP_LATIN = {"hr", "it", "ip", "email", "excel", "word", "pdf", "online", "zoom", "covid"}

# Letter pairs that are rare in English but typical of Russian typed on a QWERTY keyboard
# (e.g. "j" stands for "о", "f" for "а"). Fallback for words missing from the corpus vocabulary.
QWERTY_ANOMALY = re.compile(
    r"(?:[a-z]j|j[a-z]|w[bdfgjkmnpstvwxz]|f[wbzqdgkvz]|[ftebjs]q|q[jnt]"
    r"|[bcdfghjklmnpqrstvwx]z|^ds|^xk|^ghj|^vtn|km|mf)",
    re.IGNORECASE,
)


class LayoutFixer:
    def __init__(self, vocabulary: set[str]):
        """vocabulary — stemmed words of the corpus (see vocabulary_from_texts)."""
        self.vocabulary = vocabulary

    def fix(self, query: str) -> str:
        parts = re.split(r"(\s+)", query)
        fixed = [self._fix_token(p) for p in parts]
        latin = [i for i, p in enumerate(parts) if _is_latin_word(p)]
        flipped = [i for i in latin if fixed[i] != parts[i]]
        # Query-level context: if most Latin words turned out to be Russian typed on the wrong layout,
        # the whole query was typed that way — flip the rest too. This catches one-letter words
        # ("d" -> "в", "z" -> "я") and words missing from the corpus ("pfhgkfne" -> "зарплату").
        if flipped and 2 * len(flipped) >= len(latin):
            for i in latin:
                if fixed[i] == parts[i] and _strip(parts[i]).lower() not in KEEP_LATIN:
                    fixed[i] = _flip(parts[i])
        return "".join(fixed)

    def _fix_token(self, token: str) -> str:
        if not token or token.isspace() or re.search(r"[а-яё]", token, re.IGNORECASE):
            return token  # already Cyrillic (or whitespace)
        pre, core, post = _split_boundaries(token)
        if len(core) < 2 or core.lower() in KEEP_LATIN or core.isdigit():
            return token
        flipped = core.translate(QWERTY_TO_JCUKEN)
        if not any(c in CYRILLIC_VOWELS for c in flipped.lower()):
            return token  # no vowels after flipping: not a Russian word

        stems = tokenize(flipped)
        # Short function words ("мне", "или") are dropped by the tokenizer, so check them against the stopword list.
        is_stopword = flipped.lower().replace("ё", "е") in STOPWORDS
        in_vocabulary = is_stopword or (bool(stems) and all(s in self.vocabulary for s in stems))
        has_layout_keys = any(c in LAYOUT_PUNCTUATION for c in core)
        looks_like_qwerty_russian = len(core) >= 4 and QWERTY_ANOMALY.search(core) is not None
        if in_vocabulary or has_layout_keys or looks_like_qwerty_russian:
            return _flip(token)
        return token


def _flip(token: str) -> str:
    pre, core, post = _split_boundaries(token)
    return pre + core.translate(QWERTY_TO_JCUKEN) + post.translate(QWERTY_TO_JCUKEN)


def _strip(token: str) -> str:
    return _split_boundaries(token)[1]


def _is_latin_word(token: str) -> bool:
    core = _strip(token)
    return bool(core) and re.search(r"[a-z]", core, re.IGNORECASE) is not None and not re.search(
        r"[а-яё]", core, re.IGNORECASE
    )


def _split_boundaries(token: str) -> tuple[str, str, str]:
    """Separate leading/trailing sentence punctuation; layout keys like ',' '.' ';' stay in the core."""
    start, end = 0, len(token)
    while start < end and token[start] in BOUNDARY_CHARS:
        start += 1
    while end > start and token[end - 1] in BOUNDARY_CHARS | {"&"}:
        end -= 1
    return token[:start], token[start:end], token[end:]


def vocabulary_from_texts(texts: list[str]) -> set[str]:
    return {stem for text in texts for stem in tokenize(text)}
