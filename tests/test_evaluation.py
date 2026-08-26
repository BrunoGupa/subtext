import json
from pathlib import Path

import pytest

from reel_query.evaluation import GoldenQuestion, cited_line_ids, load_golden, parse_strategy


def test_parse_strategy_labels():
    assert parse_strategy("line") == ("line", 1)
    assert parse_strategy("window5") == ("window", 5)
    with pytest.raises(ValueError):
        parse_strategy("window")


def test_cited_line_ids_reads_the_forms_an_answer_actually_uses():
    text = "See line 42 and line_id 7, plus line #103."
    assert cited_line_ids(text) == [7, 42, 103]


def test_the_shipped_golden_set_meets_the_scoped_floor():
    questions = load_golden()
    assert len(questions) >= 30, "the CV claim needs at least 30 labelled questions"
    assert sum(1 for q in questions if q.is_empty_case) >= 3, "need expected-empty cases"
    assert len({q.id for q in questions}) == len(questions), "duplicate question ids"
    for q in questions:
        assert q.question.strip()
        assert q.kind in {"lookup", "semantic", "empty"}


def test_golden_set_ids_point_at_real_lines():
    from reel_query.ingest.sample import iter_lines

    corpus = {line.line_id: line for line in iter_lines()}
    for question in load_golden():
        for line_id in question.expected_line_ids:
            assert line_id in corpus, f"{question.id} cites a line that does not exist: {line_id}"
        if question.character:
            assert all(
                corpus[i].character == question.character for i in question.expected_line_ids
            ), f"{question.id} filters to {question.character} but expects other speakers' lines"
        if question.season is not None:
            assert all(
                corpus[i].season == question.season for i in question.expected_line_ids
            ), f"{question.id} filters to season {question.season} but expects other seasons"


def test_empty_case_questions_expect_nothing():
    for question in load_golden():
        if question.kind == "empty":
            assert question.expected_line_ids == []


class _State(dict):
    pass


class _Ctx:
    def __init__(self):
        self.state = _State()


def test_line_ids_are_harvested_from_the_mcp_result_shape():
    from reel_query.agent.tools import RETRIEVED_KEY, record_line_ids_from_rows

    ctx = _Ctx()
    payload = {"columns": ["line_id", "text"], "rows": [[9, "a"], [49, "b"]]}
    assert record_line_ids_from_rows(ctx, payload) == 2
    assert ctx.state[RETRIEVED_KEY] == [9, 49]


def test_line_ids_are_harvested_when_the_result_arrives_json_encoded():
    import json

    from reel_query.agent.tools import RETRIEVED_KEY, record_line_ids_from_rows

    ctx = _Ctx()
    payload = {"content": [{"text": json.dumps({"columns": ["line_id"], "rows": [[101]]})}]}
    record_line_ids_from_rows(ctx, payload)
    assert ctx.state[RETRIEVED_KEY] == [101]


def test_harvesting_an_unrecognised_shape_is_a_no_op_not_an_error():
    from reel_query.agent.tools import record_line_ids_from_rows

    ctx = _Ctx()
    assert record_line_ids_from_rows(ctx, {"error": "boom"}) == 0
    assert record_line_ids_from_rows(ctx, None) == 0
    assert record_line_ids_from_rows(None, {"columns": ["line_id"], "rows": [[1]]}) == 0
