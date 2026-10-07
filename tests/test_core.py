"""Fast offline tests: no network, no API key."""

from rag.corpus import Chunk, chunk_article, group_parts, load_chunks
from rag.layout import LayoutFixer, vocabulary_from_texts
from rag.retrieval import BM25Retriever, rrf
from rag.text import tokenize


def test_list_items_are_glued_to_their_part():
    paragraphs = ["Договор может быть расторгнут в случаях:", "1) ликвидации организации;", "2) сокращения штата;", "Вторая часть."]
    assert group_parts(paragraphs) == [
        "Договор может быть расторгнут в случаях:\n1) ликвидации организации;\n2) сокращения штата;",
        "Вторая часть.",
    ]


def test_long_article_is_split_without_losing_part_numbers():
    record = {"article": "1", "title": "t", "chapter": "c", "parts": ["а" * 900, "б" * 900, "в" * 100]}
    chunks = chunk_article(record, max_chars=1500)
    assert [(c["part_start"], c["part_end"]) for c in chunks] == [(1, 1), (2, 3)]


def test_citation_format():
    one = Chunk(0, "115", "t", "c", 1, 1, ("x",))
    many = Chunk(0, "81", "t", "c", 2, 4, ("x", "y", "z"))
    assert one.citation == "ст. 115, ч. 1"
    assert many.citation == "ст. 81, ч. 2–4"
    assert many.text.splitlines()[2].startswith("(ч. 2)")


def test_stemming_matches_word_forms():
    assert tokenize("отпуска") == tokenize("отпуском") == tokenize("отпуск")
    assert tokenize("и в на") == []


def test_rrf_rewards_agreement_between_rankings():
    fused = [doc for doc, _ in rrf([[1, 2, 3], [3, 2, 1], [2, 4]])]
    assert fused[0] == 2  # second in two lists and first in one beats being first in one list only


def test_bm25_finds_vacation_article_on_real_corpus():
    chunks = load_chunks()
    hits = BM25Retriever(chunks).search("продолжительность ежегодного основного оплачиваемого отпуска", k=3)
    assert "115" in {chunks[i].article for i, _ in hits}


def _fixer() -> LayoutFixer:
    return LayoutFixer(vocabulary_from_texts([c.text for c in load_chunks()]))


def test_layout_fixer_repairs_whole_query_typed_in_english_layout():
    assert _fixer().fix("crjkmrj lytq jngecrf vyt gjkj;tyj&") == "сколько дней отпуска мне положено?"


def test_layout_fixer_repairs_only_wrong_words_in_mixed_query():
    assert _fixer().fix("сколько дней jngecrf положено?") == "сколько дней отпуска положено?"


def test_layout_fixer_keeps_correct_text_and_latin_terms():
    fixer = _fixer()
    assert fixer.fix("Можно ли работать удалённо?") == "Можно ли работать удалённо?"
    assert fixer.fix("Что должен делать HR при увольнении?") == "Что должен делать HR при увольнении?"


def test_layout_fixer_uses_query_context_for_short_and_unknown_words():
    fixer = _fixer()
    assert fixer.fix("Crjkmrj xfcjd d ytltk. z ljk;ty hf,jnfnm&") == "Сколько часов в неделю я должен работать?"
    assert fixer.fix("Rfr xfcnj dslfdfnm pfhgkfne&") == "Как часто выдавать зарплату?"
