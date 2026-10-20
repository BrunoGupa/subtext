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
    if s.db_backend == "chdb":
        return _chdb_client()
    return clickhouse_connect.get_client(
        host=s.ch_host,
        port=s.ch_port,
        username=s.ch_user,
        password=s.ch_password,
        database=database if database is not None else s.ch_database,
        secure=s.ch_secure,
    )


# --------------------------------------------------------------------------- chDB


#: The tables the web page reads, and nothing else. The 103.6M-row `aligned_lines` they were
#: built from is not needed to answer a line, which is what lets the demo fit in one image.
RUNTIME_TABLES: dict[str, str] = {
    "mx_corpus": """
        CREATE TABLE IF NOT EXISTS subtext.mx_corpus
        (pair_id UInt64, blk UInt32, doc_id UInt32, en String, es String)
        ENGINE = MergeTree ORDER BY pair_id
    """,
    # No HNSW index: 269,869 vectors are scanned in full in well under a second, and an
    # exact scan cannot miss a neighbour the way an approximate graph can.
    "mx_embeddings": """
        CREATE TABLE IF NOT EXISTS subtext.mx_embeddings
        (text String, embedding Array(Float32))
        ENGINE = MergeTree ORDER BY text
    """,
    "phrase_index": """
        CREATE TABLE IF NOT EXISTS subtext.phrase_index
        (ng String, n UInt8, support UInt32, ex_pair_id UInt64)
        ENGINE = MergeTree ORDER BY (n, ng)
    """,
    "address_tags": """
        CREATE TABLE IF NOT EXISTS subtext.address_tags
        (es String, model LowCardinality(String), form LowCardinality(String),
         tagged DateTime DEFAULT now())
        ENGINE = ReplacingMergeTree(tagged) ORDER BY (es, model)
    """,
    "web_cache": """
        CREATE TABLE IF NOT EXISTS subtext.web_cache
        (line String, payload String, created DateTime DEFAULT now())
        ENGINE = ReplacingMergeTree(created) ORDER BY line
    """,
}


def _literal(value: Any) -> str:
    """A value as ClickHouse SQL literal text."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_literal(v) for v in value) + "]"
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def _param(value: Any) -> str:
    """A bound parameter in the text form the server parses. Strings go bare (the type in
    `{name:String}` already says what they are); arrays need their elements quoted."""
    if isinstance(value, (list, tuple)):
        return _literal(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


class _Result:
    def __init__(self, column_names: list[str], result_rows: list[tuple]):
        self.column_names = column_names
        self.result_rows = result_rows


class ChdbClient:
    """The slice of `clickhouse_connect`'s client this project uses, over an in-process chDB
    session. One session per process and one statement at a time: the web layer queries
    from a thread pool, and a lock is cheaper than finding out what chDB does without one.
    """

    _WIDE = ("UInt64", "Int64", "UInt128", "Int128", "UInt256", "Int256")

    def __init__(self, path: str):
        import threading

        from chdb import session

        self._session = session.Session(path)
        self._lock = threading.Lock()
        self._session.query("CREATE DATABASE IF NOT EXISTS subtext")

    def _run(self, sql: str, fmt: str, parameters: dict | None = None):
        params = {k: _param(v) for k, v in (parameters or {}).items()} or None
        with self._lock:
            self._session.query("USE subtext")
            return self._session.query(sql, fmt, params=params)

    def query(self, sql: str, parameters: dict | None = None, settings: dict | None = None):
        import json

        result = self._run(sql, "JSONCompact", parameters)
        text = result.data() if result is not None else ""
        if not text.strip():
            return _Result([], [])
        payload = json.loads(text)
        meta = payload.get("meta", [])
        # JSONCompact quotes 64-bit integers so JavaScript cannot round them; Python can
        # hold them, and callers do arithmetic on some and compare others directly.
        wide = [i for i, m in enumerate(meta) if m["type"].startswith(self._WIDE)]
        rows = []
        for row in payload.get("data", []):
            if wide:
                row = list(row)
                for i in wide:
                    if isinstance(row[i], str):
                        row[i] = int(row[i])
            rows.append(tuple(row))
        return _Result([m["name"] for m in meta], rows)

    def command(self, sql: str, settings: dict | None = None) -> None:
        self._run(sql, "CSV")

    def insert(self, table: str, data, column_names: Sequence[str]) -> None:
        if not data:
            return
        values = ",".join("(" + ",".join(_literal(v) for v in row) + ")" for row in data)
        self._run(f"INSERT INTO {table} ({', '.join(column_names)}) VALUES {values}", "CSV")


_chdb_singleton: ChdbClient | None = None


def _chdb_client() -> ChdbClient:
    global _chdb_singleton
    if _chdb_singleton is None:
        _chdb_singleton = ChdbClient(settings().chdb_path)
    return _chdb_singleton


def build_chdb(parquet_dir: str, path: str | None = None, log=print) -> dict[str, int]:
    """Create the runtime tables in a chDB directory and fill them from Parquet files named
    after them (`mx_corpus.parquet`, ...). A missing file leaves its table empty, which is
    right for the two caches."""
    from pathlib import Path

    ch = ChdbClient(path or settings().chdb_path)
    counts: dict[str, int] = {}
    for table, ddl in RUNTIME_TABLES.items():
        ch.command(f"DROP TABLE IF EXISTS subtext.{table}")
        ch.command(ddl)
        source = Path(parquet_dir).resolve() / f"{table}.parquet"
        if source.exists():
            ch.command(f"INSERT INTO subtext.{table} SELECT * FROM file('{source}', Parquet)")
            ch.command(f"OPTIMIZE TABLE subtext.{table} FINAL")
        counts[table] = ch.query(f"SELECT count() FROM subtext.{table}").result_rows[0][0]
        log(f"{table}: {counts[table]:,} rows")
    return counts


def export_runtime(parquet_dir: str, log=print) -> None:
    """Write the runtime tables, from whichever backend is configured, as Parquet."""
    from pathlib import Path

    out = Path(parquet_dir)
    out.mkdir(parents=True, exist_ok=True)
    ch = client()
    for table in RUNTIME_TABLES:
        target = out / f"{table}.parquet"
        if isinstance(ch, ChdbClient):
            ch.command(f"SELECT * FROM subtext.{table} INTO OUTFILE '{target.resolve()}' "
                       "TRUNCATE FORMAT Parquet")
        else:
            try:
                data = ch.raw_query(f"SELECT * FROM {table} FORMAT Parquet")
            except Exception as exc:  # a cache table that was never created
                log(f"{table}: skipped ({str(exc).splitlines()[0][:80]})")
                continue
            target.write_bytes(data)
        log(f"{table}: {target.stat().st_size / 1e6:.1f} MB")


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
