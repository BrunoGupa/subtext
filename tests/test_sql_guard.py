import pytest

from reel_query.sql_guard import UnsafeSQL, validate


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count() FROM lines",
        "WITH c AS (SELECT 1 AS x) SELECT x FROM c",
        "select character, count() from lines group by character",
        "SELECT text FROM lines WHERE text LIKE '%drop table%'",
        "SELECT 1 -- insert into lines",
        "SELECT count() FROM lines;",
    ],
)
def test_reads_are_allowed(sql):
    assert validate(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE lines",
        "INSERT INTO lines VALUES (1)",
        "ALTER TABLE lines DELETE WHERE 1",
        "SELECT 1; DROP TABLE lines",
        "TRUNCATE TABLE lines",
        "SET max_threads = 1",
        "SELECT * FROM system.tables",
        "",
        "   ",
    ],
)
def test_writes_and_multi_statements_are_rejected(sql):
    with pytest.raises(UnsafeSQL):
        validate(sql)


def test_keyword_hidden_in_a_comment_does_not_pass_as_sql():
    # The comment is stripped, so the statement is judged on its real content.
    assert validate("SELECT 1 /* drop table lines */")
