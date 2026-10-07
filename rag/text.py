"""Tokenization for lexical search.

Russian is highly inflected: "отпуск", "отпуска", "отпуском" are one word for a human but three
different tokens for BM25. A stemmer cuts endings ("отпуск" for all three), which is cheap and
fixes most of the mismatch between how employees ask and how the law is written.
"""

import re

import snowballstemmer

_WORD_RE = re.compile(r"[а-яa-z0-9]+")
_stemmer = snowballstemmer.stemmer("russian")

# Frequent function words carry no meaning for search.
STOPWORDS = frozenset(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне "
    "было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до "
    "вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя "
    "их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого "
    "какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно "
    "при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве "
    "три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда "
    "конечно всю между ли сколько какие каков какова какое".split()
)


def tokenize(text: str) -> list[str]:
    words = _WORD_RE.findall(text.lower().replace("ё", "е"))
    return _stemmer.stemWords([w for w in words if w not in STOPWORDS])
