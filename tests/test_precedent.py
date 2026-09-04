"""The Precedent record and the consensus signal it exposes.

These are pure unit tests: no ClickHouse, no model. The retrieval query itself is
exercised by hand against the live corpus, not here.
"""

from subtext.retrieval import Precedent


def make(spanish="Órale.", times=3, english_times=4, similarity=0.9, english="Hurry up."):
    return Precedent(english=english, spanish=spanish, pair_id=42, doc_id=7,
                     times=times, english_times=english_times,
                     similarity=similarity, exact=False)


def test_citation_names_the_row_and_the_document():
    assert make().citation == "pair_id 42, document 7"


def test_consensus_is_the_share_of_translators_agreeing():
    assert make(times=3, english_times=4).consensus == 0.75


def test_a_lone_reading_among_many_scores_low():
    """This is the shape of a misaligned row: one odd reading against several agreeing."""
    assert make(times=1, english_times=11).consensus < 0.1


def test_a_single_example_is_full_consensus_not_zero():
    """One row is weak evidence, but it is not *contradicted* evidence."""
    assert make(times=1, english_times=1).consensus == 1.0


def test_consensus_never_divides_by_zero():
    assert make(times=0, english_times=0).consensus == 0.0


def test_precedent_is_immutable():
    """Retrieved evidence should not be editable in place by a caller or an agent."""
    import dataclasses
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        make().spanish = "something else"


def test_vector_index_granularity_spans_a_whole_part():
    """The trap that cost an afternoon: GRANULARITY 1 gives one HNSW graph per 8,192-row
    granule, and the search then returns confident nonsense with no error. A vector index
    wants one graph over the whole part, so this must stay large."""
    from subtext.embeddings import VECTOR_INDEX_GRANULARITY

    assert VECTOR_INDEX_GRANULARITY >= 1_000_000
