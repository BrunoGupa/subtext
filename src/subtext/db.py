"""ClickHouse connection and the DDL for the corpus.

The schema is deliberately narrow: one table of dialogue lines, one table of chunks
(a chunk is what actually gets embedded, and chunk size is a *parameter* so the
sweep in `evals/` can vary it), and one table of schema documentation that is itself
embedded so text-to-SQL prompts only ever see the relevant columns.
"""

from __future__ import annotations

from typing import Any, Sequence

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from .config import settings


def client(database: str | None = None) -> Client:
    s = settings()
    return clickhouse_connect.get_client(
        host=s.ch_host,
        port=s.ch_port,
        username=s.ch_user,
        password=s.ch_password,
        database=database if database is not None else s.ch_database,
        secure=s.ch_secure,
    )


def query_rows(sql: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as dicts."""
    result = client().query(sql, parameters=parameters or {})
    cols = result.column_names
    return [dict(zip(cols, row)) for row in result.result_rows]


DDL: Sequence[str] = (
    """
    CREATE TABLE IF NOT EXISTS lines
    (
        line_id      UInt64,
        title        LowCardinality(String),
        title_id     LowCardinality(String),
        season       UInt16,
        episode      UInt16,
        episode_title String,
        scene        UInt16,
        line_no      UInt32,
        character    LowCardinality(String),
        speaks_to    String,
        timecode     String,
        start_ms     UInt32,
        end_ms       UInt32,
        text         String
    )
    ENGINE = MergeTree
    ORDER BY (title_id, season, episode, line_no)
    """,
    """
    CREATE TABLE IF NOT EXISTS line_chunks
    (
        chunk_id     UInt64,
        strategy     LowCardinality(String),
        window_size  UInt8,
        line_id      UInt64,
        line_ids     Array(UInt64),
        title_id     LowCardinality(String),
        season       UInt16,
        episode      UInt16,
        character    LowCardinality(String),
        start_ms     UInt32,
        text         String,
        embedding    Array(Float32)
    )
    ENGINE = MergeTree
    ORDER BY (strategy, title_id, season, episode, chunk_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS aligned_lines
    (
        pair_id      UInt64,
        corpus       LowCardinality(String),
        lang_pair    LowCardinality(String),
        source_lang  LowCardinality(String),
        target_lang  LowCardinality(String),
        source_text  String,
        target_text  String,
        INDEX idx_source source_text TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4,
        INDEX idx_target target_text TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4
    )
    ENGINE = MergeTree
    ORDER BY (lang_pair, pair_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_docs
    (
        doc_id       UInt64,
        table_name   LowCardinality(String),
        column_name  String,
        data_type    String,
        description  String,
        doc_text     String,
        embedding    Array(Float32)
    )
    ENGINE = MergeTree
    ORDER BY (table_name, doc_id)
    """,
)


def init_db() -> None:
    """Create the database and every table. Idempotent."""
    s = settings()
    admin = client(database="default")
    admin.command(f"CREATE DATABASE IF NOT EXISTS {s.ch_database}")
    c = client()
    for statement in DDL:
        c.command(statement)


def drop_all() -> None:
    c = client()
    for table in ("line_chunks", "lines", "aligned_lines", "schema_docs"):
        c.command(f"DROP TABLE IF EXISTS {table}")
