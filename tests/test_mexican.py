"""The detection lexicons and the SQL built from them.

These pin the things that would silently change the corpus if edited carelessly: the size
and content of the word lists, whole-word matching, and the round trip of the shipped
boundary index.
"""

import pytest

from subtext.ingest import mexican as m


def test_lexicon_sizes_are_pinned():
    """Widening a list is fine, but it changes the corpus - so it has to be deliberate.

    99 -> 93 on 2026-09-07, and narrowing turned out to matter far more than widening ever
    had: `simon`, `sepa`, `mande`, `chin` and `feria` were not markers at all, and removing
    them plus two judgement calls halved the corpus while more than doubling its marker
    density. See CORPUS.md 6.3.
    """
    assert (len(m.MEXICAN), len(m.PENINSULAR), len(m.LATIN_AMERICAN), len(m.OTHER_LATAM)) \
        == (93, 66, 34, 28)


def test_the_removed_entries_stay_removed():
    """Each of these was read in context before it was cut, and each would come back the
    moment somebody widened the lexicon by intuition instead of by reading."""
    for word in ("simon", "sepa", "mande", "chin", "feria", "lana", "huevon"):
        assert word not in m.MEXICAN, word


def test_a_mid_sentence_capital_is_not_a_marker():
    """`normalised` strips capitalised words that do not open a sentence, because they are
    proper names: Chava in Fiddler on the Roof, Morra in Limitless, Gacha in Narcos."""
    sql = m.normalised("c")
    assert "upperUTF8" in sql          # all-caps subtitle lines are exempt
    assert "replaceRegexpAll" in sql


def test_lexicons_are_accent_stripped_and_lowercase():
    for name in ("MEXICAN", "PENINSULAR", "LATIN_AMERICAN", "OTHER_LATAM", "GOLD"):
        for word in getattr(m, name):
            assert word == word.lower(), (name, word)
            assert not set(word) & set("áéíóúñüÁÉÍÓÚÑÜ"), (name, word)


def test_the_two_registers_do_not_overlap():
    """A word claimed for both Mexico and Spain would make the ratio test meaningless."""
    assert not (set(m.MEXICAN) & set(m.PENINSULAR))


def test_gold_is_a_subset_of_the_detector_lexicon():
    """Which is exactly why it is not a held-out set - the docstring says so, pin it."""
    single = {w for w in m.GOLD if " " not in w}
    assert single <= set(m.MEXICAN)


def test_words_are_space_padded_for_whole_word_matching():
    """Unpadded, ' vale ' would fire inside 'equivale' and the peninsular count would lie."""
    sql = m.any_of(["vale"])
    assert "' vale '" in sql
    assert "'vale'" not in sql


def test_apostrophes_in_a_word_are_escaped_for_sql():
    assert "'' " in m.any_of(["o'brien"]) or "o''brien" in m.any_of(["o'brien"])


def test_normalised_pads_and_strips_accents():
    sql = m.normalised("es")
    assert sql.startswith("concat(' '")
    assert "aeiounuaeiounu" in sql
    assert "[^a-z0-9]+" in sql


def test_seed_threshold_matches_the_published_build():
    sql = m.seed_sql("win_scores")
    assert "mx >= 10" in sql
    assert "mx >= 2 * es" in sql
    assert "intDiv(b, 40)" in sql  # 1000-line windows from 25-line buckets


@pytest.mark.parametrize("builder", ["win_scores_sql", "gold_sql"])
def test_generated_sql_targets_the_named_table(builder):
    sql = getattr(m, builder)("some_table")
    assert sql.startswith("INSERT INTO some_table")
    assert "GROUP BY b" in sql


def test_statements_ignores_semicolons_inside_comments():
    script = "-- a note; with a semicolon\nSELECT 1;\nSELECT 2;\n"
    assert m.statements(script) == ["SELECT 1", "SELECT 2"]


def test_index_round_trip(tmp_path):
    p = tmp_path / "idx.tsv"
    p.write_text("doc_id\tlo\thi\tn_lines\tmx\tes\tother\n"
                 "1\t100\t199\t100\t5\t1\t0\n"
                 "2\t500\t799\t300\t9\t0\t0\n", encoding="utf-8")
    assert m.read_index(p) == [(100, 199), (500, 799)]


def test_index_reader_uses_column_names_not_positions(tmp_path):
    """So a future extra column cannot silently shift the ranges."""
    p = tmp_path / "idx.tsv"
    p.write_text("doc_id\tnote\tlo\thi\n1\tx\t7\t9\n", encoding="utf-8")
    assert m.read_index(p) == [(7, 9)]
