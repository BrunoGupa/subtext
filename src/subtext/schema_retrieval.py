"""Schema retrieval — grounding text-to-SQL in the *relevant* part of the schema.

Pasting an entire catalogue into a prompt stops working the moment the schema is
wide, and it actively hurts: irrelevant columns are what a model hallucinates joins
on. So each column gets a one-line description, the descriptions are embedded with
the same local model as the dialogue, and only the columns nearest the question are
rendered into the SQL-generation prompt.

The descriptions live here, in code, next to the DDL they describe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .db import client
from .embeddings import embed, embed_one

# (table, column, type, description)
COLUMN_DOCS: tuple[tuple[str, str, str, str], ...] = (
    ("lines", "line_id", "UInt64", "Unique id of a single spoken line of dialogue. The join key used by every retrieval path and the unit the golden set is labelled in."),
    ("lines", "title", "String", "Human-readable title of the film or series the line comes from, e.g. 'The Lighthouse Contract'."),
    ("lines", "title_id", "String", "Stable identifier for the title, IMDb-style. Use this to filter to one show rather than matching on the title text."),
    ("lines", "season", "UInt16", "Season number the line was spoken in. 0 for films, which have no seasons. Use for per-season counts and 'across N seasons' questions."),
    ("lines", "episode", "UInt16", "Episode number within the season. 0 for films."),
    ("lines", "episode_title", "String", "Title of the episode the line appears in."),
    ("lines", "scene", "UInt16", "Scene number within the episode, for grouping consecutive dialogue that happens in one place."),
    ("lines", "line_no", "UInt32", "Position of the line within its episode, counting from 1. Use to order dialogue or to find what was said just before or after a line."),
    ("lines", "character", "String", "Name of the character who speaks the line. Use for 'what did X say' and for per-character aggregation."),
    ("lines", "speaks_to", "String", "Name of the character being addressed, when known. Use for questions about what one character said to another."),
    ("lines", "timecode", "String", "Human-readable subtitle timestamp of the line, HH:MM:SS,mmm. Show this to the user; do not sort or compare on it."),
    ("lines", "start_ms", "UInt32", "Start of the line in milliseconds from the beginning of the episode. Use this for ordering by time and for time-window filters."),
    ("lines", "end_ms", "UInt32", "End of the line in milliseconds from the beginning of the episode. Duration is end_ms - start_ms."),
    ("lines", "text", "String", "The spoken words of the line. Keyword matching works here, but meaning-based questions must go through vector search over line_chunks."),
    ("line_chunks", "chunk_id", "UInt64", "Unique id of an embedded chunk. A chunk is the unit that is retrieved by vector similarity."),
    ("line_chunks", "strategy", "String", "Which chunking strategy produced this chunk: 'line' (one line per chunk) or 'window' (N consecutive lines). Always filter on one strategy."),
    ("line_chunks", "window_size", "UInt8", "Number of dialogue lines covered by the chunk. 1 for the 'line' strategy. Always filter on one window_size alongside strategy."),
    ("line_chunks", "line_id", "UInt64", "The anchor line of the chunk: the first line the chunk covers."),
    ("line_chunks", "line_ids", "Array(UInt64)", "Every line_id the chunk covers. ARRAY JOIN this and join to lines.line_id to resolve a semantic hit back to individual dialogue lines."),
    ("line_chunks", "title_id", "String", "Title the chunk belongs to, denormalised from lines so a vector search can be scoped to one show without a join."),
    ("line_chunks", "season", "UInt16", "Season the chunk belongs to, denormalised from lines."),
    ("line_chunks", "episode", "UInt16", "Episode the chunk belongs to, denormalised from lines."),
    ("line_chunks", "character", "String", "Character who speaks the anchor line of the chunk. For window chunks the chunk may span several speakers; join through line_ids when exact attribution matters."),
    ("line_chunks", "start_ms", "UInt32", "Start time of the chunk's anchor line, in milliseconds."),
    ("line_chunks", "text", "String", "The chunk's text as it was embedded: 'CHARACTER: line' for the line strategy, several such lines joined by newlines for the window strategy."),
    ("line_chunks", "embedding", "Array(Float32)", "Embedding of the chunk text. Rank semantic matches with cosineDistance(embedding, <query vector>) ASC; smaller distance is more similar."),
    ("schema_docs", "doc_id", "UInt64", "Unique id of a schema documentation entry."),
    ("schema_docs", "table_name", "String", "Table the documented column belongs to."),
    ("schema_docs", "column_name", "String", "Name of the documented column."),
    ("schema_docs", "data_type", "String", "ClickHouse type of the documented column."),
    ("schema_docs", "description", "String", "Natural-language description of what the column means and when to use it."),
    ("schema_docs", "doc_text", "String", "The embedded form of the documentation entry: table, column, type and description in one string."),
    ("schema_docs", "embedding", "Array(Float32)", "Embedding of doc_text, used to retrieve the relevant slice of the schema for a question."),
)

TABLE_NOTES: dict[str, str] = {
    "lines": "One row per spoken line of dialogue. The structured corpus: who said what, in which title, season, episode and at what timecode.",
    "line_chunks": "One row per embedded chunk of dialogue, with its vector. The semantic index over `lines`. Always constrain to a single (strategy, window_size) pair.",
    "schema_docs": "Documentation of the schema itself, embedded so that only the relevant columns are put in front of the SQL generator.",
}


@dataclass(frozen=True)
class SchemaDoc:
    table_name: str
    column_name: str
    data_type: str
    description: str
    distance: float = 0.0


def _doc_text(table: str, column: str, data_type: str, description: str) -> str:
    return f"Table {table}. Column {column} ({data_type}). {TABLE_NOTES.get(table, '')} {description}"


def build_schema_docs() -> int:
    """Embed every column description and persist it. Idempotent — replaces the table."""
    texts = [_doc_text(*doc) for doc in COLUMN_DOCS]
    vectors = embed(texts)
    rows = [
        [i, table, column, data_type, description, text, vector]
        for i, ((table, column, data_type, description), text, vector) in enumerate(
            zip(COLUMN_DOCS, texts, vectors)
        )
    ]
    c = client()
    c.command("TRUNCATE TABLE IF EXISTS schema_docs")
    c.insert(
        "schema_docs",
        rows,
        column_names=[
            "doc_id", "table_name", "column_name", "data_type",
            "description", "doc_text", "embedding",
        ],
    )
    return len(rows)


def retrieve_schema(question: str, *, k: int = 12) -> list[SchemaDoc]:
    """The `k` column descriptions nearest the question."""
    result = client().query(
        """
        SELECT table_name, column_name, data_type, description,
               cosineDistance(embedding, {q:Array(Float32)}) AS distance
        FROM schema_docs
        WHERE length(embedding) > 0
        ORDER BY distance ASC
        LIMIT {k:UInt32}
        """,
        parameters={"q": embed_one(question), "k": k},
    )
    return [
        SchemaDoc(
            table_name=row[0],
            column_name=row[1],
            data_type=row[2],
            description=row[3],
            distance=float(row[4]),
        )
        for row in result.result_rows
    ]


def render_schema_prompt(docs: Iterable[SchemaDoc]) -> str:
    """Render retrieved columns as a compact DDL-ish block for the SQL prompt."""
    by_table: dict[str, list[SchemaDoc]] = {}
    for doc in docs:
        by_table.setdefault(doc.table_name, []).append(doc)

    blocks: list[str] = []
    for table, columns in by_table.items():
        lines = [f"-- {TABLE_NOTES.get(table, '')}".rstrip(), f"TABLE {table} ("]
        lines += [
            f"    {c.column_name} {c.data_type},  -- {c.description}" for c in columns
        ]
        lines.append(")")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def schema_context(question: str, *, k: int = 12) -> dict[str, Any]:
    docs = retrieve_schema(question, k=k)
    return {
        "columns_retrieved": len(docs),
        "columns_available": len(COLUMN_DOCS),
        "tables": sorted({d.table_name for d in docs}),
        "prompt_block": render_schema_prompt(docs),
    }
