"""The index build and the lookup path must cut words identically, or every
apostrophe-carrying phrase misses the index silently. These tests pin that."""

import re

import pytest

from subtext.tokenizer import (
    MAX_N,
    WORD_PATTERN,
    clickhouse_tokens_sql,
    ngrams,
    normalize,
    phrase_index_sql,
    phrases,
    tokens,
)


def test_internal_apostrophe_stays_inside_the_word():
    assert tokens("What's up, dude?") == ["what's", "up", "dude"]


@pytest.mark.parametrize(
    "text,contraction,collides_with",
    [("We're late", "we're", "were"),
     ("I'll go", "i'll", "ill"),
     ("He'll pay", "he'll", "hell"),
     ("It's mine", "it's", "its"),
     ("She'd know", "she'd", "shed"),
     ("I can't", "can't", "cant")],
)
def test_contractions_never_collide_with_a_different_word(text, contraction, collides_with):
    """The bug this module exists to prevent: `we're` and `were` sharing one index key."""
    assert contraction in tokens(text)
    assert collides_with not in tokens(text)


def test_typographic_apostrophes_fold_to_ascii():
    """OPUS normalized quotes; a user's .srt will not."""
    for variant in "’ʼ′`´":
        assert tokens(f"what{variant}s up") == ["what's", "up"]


def test_trailing_apostrophe_is_not_part_of_the_word():
    assert tokens("the boys' club") == ["the", "boys", "club"]
    assert tokens("'Tis nothing") == ["tis", "nothing"]


def test_normalize_lowercases_and_leaves_words_joinable():
    assert normalize("What’s UP") == "what's up"


def test_digits_and_punctuation_are_dropped():
    assert tokens("Room 237, now!") == ["room", "now"]


def test_ngrams_slide_by_one():
    assert list(ngrams(["a", "b", "c"], 2)) == ["a b", "b c"]
    assert list(ngrams(["a", "b"], 3)) == []


def test_ngrams_rejects_n_below_one():
    with pytest.raises(ValueError):
        list(ngrams(["a"], 0))


def test_phrases_covers_every_length_up_to_max_n():
    got = list(phrases("what's up dude", max_n=2))
    assert got == ["what's", "up", "dude", "what's up", "up dude"]


def test_a_contraction_costs_one_token_not_two():
    """Why we keep the token whole rather than splitting Penn-Treebank style: an n=4
    window must cover four real words, not three words and a clitic."""
    assert len(tokens("what's up dude now")) == MAX_N


def test_clickhouse_expression_escapes_the_apostrophe_for_sql():
    sql = clickhouse_tokens_sql("en")
    assert sql == "extractAll(lowerUTF8(en), '[a-z]+(?:''[a-z]+)?')"
    # Doubling is SQL string escaping only; unescaping recovers the Python pattern.
    inner = re.fullmatch(r"extractAll\(lowerUTF8\(en\), '(.*)'\)", sql).group(1)
    assert inner.replace("''", "'") == WORD_PATTERN


def test_generated_sql_builds_every_n_and_is_marked_generated():
    sql = phrase_index_sql()
    assert "GENERATED" in sql
    for n in range(1, MAX_N + 1):
        assert f"SELECT ng, {n} AS n" in sql
    assert sql.count(clickhouse_tokens_sql("en")) == MAX_N
