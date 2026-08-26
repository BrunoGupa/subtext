"""The agent's tools.

Every one of these is deterministic Python. The model decides *which* to call and
*what to ask*; it never does the retrieving, the counting or the checking itself.
That split is deliberate: a step that makes no creative decision should not be an
LLM call.

Each retrieval tool records what it returned into session state under
``retrieved_line_ids``. :func:`verify_answer` then checks the model's citations
against that record — which is what makes "the agent validates its answer against the
data before speaking" a mechanical fact rather than a claim in a prompt.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.tools import ToolContext

from .. import retrieval
from ..db import client
from ..schema_retrieval import schema_context
from ..sql_guard import READONLY_SETTINGS, MAX_ROWS, UnsafeSQL, validate

RETRIEVED_KEY = "retrieved_line_ids"
DEFAULT_STRATEGY = "line"
DEFAULT_WINDOW = 1


def _record(tool_context: ToolContext | None, line_ids: list[int]) -> None:
    if tool_context is None:
        return
    seen = set(tool_context.state.get(RETRIEVED_KEY, []))
    seen.update(int(i) for i in line_ids)
    tool_context.state[RETRIEVED_KEY] = sorted(seen)


def inspect_schema(question: str, tool_context: ToolContext = None) -> dict[str, Any]:  # type: ignore[assignment]
    """Retrieve the slice of the database schema relevant to a question.

    Call this before writing SQL. It returns only the columns that matter for this
    question, not the whole catalogue.

    Args:
        question: The user's question, in their own words.

    Returns:
        The relevant columns rendered as DDL, plus how many of the total were kept.
    """
    return schema_context(question, k=12)


def search_dialogue(
    query: str,
    k: int = 10,
    character: str = "",
    involving: str = "",
    season: int = 0,
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Find dialogue lines by meaning, using vector similarity in ClickHouse.

    Use this for anything the words in the database do not say literally — a promise
    being broken, a lie, an apology. Keyword matching will not find those.

    Args:
        query: What to look for, phrased as the meaning you want, not as SQL.
        k: How many lines to return. 10 is a good default; raise it for counting questions.
        character: Restrict to lines this character SPEAKS. Use only when the question is
            about what they said.
        involving: Restrict to lines this character speaks OR is spoken to. Use this for
            questions about what a character DID — someone breaking a promise is usually
            evidenced by another character calling them out, not by their own line.
        season: Optional season number to restrict the search to. 0 means all seasons.

    Returns:
        Ranked lines with their line_id, season, episode, character, timecode and distance.
    """
    hits = retrieval.search(
        query,
        k=k,
        strategy=DEFAULT_STRATEGY,
        window_size=DEFAULT_WINDOW,
        character=character or None,
        involving=involving or None,
        season=season or None,
    )
    _record(tool_context, [h.line_id for h in hits])
    return {
        "count": len(hits),
        "hits": [
            {
                "line_id": h.line_id,
                "season": h.season,
                "episode": h.episode,
                "character": h.character,
                "timecode": h.timecode,
                "text": h.text,
                "distance": round(h.distance, 4),
            }
            for h in hits
        ],
    }


def aggregate_semantic_matches(
    query: str,
    group_by: str = "season",
    k: int = 40,
    character: str = "",
    involving: str = "",
    max_distance: float = 0.65,
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Count semantically matching dialogue, grouped by a structured column.

    This is the hybrid path and the right tool for "how many times does X ...?".
    It runs ONE ClickHouse query in which a vector search feeds a SQL GROUP BY, so
    the counting happens over exactly the lines the semantic search returned.

    Args:
        query: The meaning to search for, e.g. "goes back on his word, breaks a promise".
        group_by: Comma-separated columns to group by: season, episode, character, title_id, episode_title.
        k: Size of the semantic candidate pool before grouping.
        character: Restrict to lines this character SPEAKS.
        involving: Restrict to lines this character speaks OR is spoken to. This is almost
            always the right filter for "how many times does X ...?", because the evidence
            for what X did is usually in someone else's line.
        max_distance: Cosine-distance ceiling for a candidate to count. Lower is stricter.

    Returns:
        One row per group with a match count, the best distance, the line_ids behind
        the count, and the SQL that produced it.
    """
    columns = tuple(c.strip() for c in group_by.split(",") if c.strip()) or ("season",)
    try:
        rows, sql = retrieval.hybrid_aggregate(
            query,
            group_by=columns,
            k=k,
            strategy=DEFAULT_STRATEGY,
            window_size=DEFAULT_WINDOW,
            character=character or None,
            involving=involving or None,
            max_distance=max_distance,
        )
    except ValueError as exc:
        return {"error": str(exc), "rows": []}

    line_ids: list[int] = []
    for row in rows:
        line_ids.extend(int(i) for i in row.get("line_ids", []))
    _record(tool_context, line_ids)

    return {
        "row_count": len(rows),
        "total_matches": sum(int(r["matches"]) for r in rows),
        "rows": [
            {
                **{c: r[c] for c in columns},
                "matches": int(r["matches"]),
                "best_distance": float(r["best_distance"]),
                "line_ids": [int(i) for i in r["line_ids"]],
                "examples": list(r["examples"]),
            }
            for r in rows
        ],
        "sql": sql,
        "note": (
            "Empty rows means nothing in the corpus was semantically close enough. "
            "Say so plainly, or retry with a differently phrased query or a higher "
            "max_distance — do not invent a count."
        ),
    }


def run_sql(sql: str, tool_context: ToolContext = None) -> dict[str, Any]:  # type: ignore[assignment]
    """Execute a read-only SELECT against the corpus.

    Use this for anything structural: filters, joins, counts, ordering by season or
    timecode. It cannot answer meaning-based questions on its own — get candidate
    lines from search_dialogue first, then aggregate them here if you need to.

    Args:
        sql: A single read-only SELECT (or WITH ... SELECT) statement.

    Returns:
        The rows, or an error explaining what was rejected and why.
    """
    try:
        statement = validate(sql)
    except UnsafeSQL as exc:
        return {"error": f"rejected: {exc}", "rows": []}

    try:
        result = client().query(statement, settings=READONLY_SETTINGS)
    except Exception as exc:  # ClickHouse errors are the model's feedback loop
        return {"error": f"clickhouse: {type(exc).__name__}: {exc}", "rows": []}

    rows = [dict(zip(result.column_names, row)) for row in result.result_rows[:MAX_ROWS]]
    if "line_id" in result.column_names:
        _record(tool_context, [int(r["line_id"]) for r in rows])
    return {
        "row_count": len(rows),
        "columns": list(result.column_names),
        "rows": rows,
        "truncated": len(result.result_rows) > MAX_ROWS,
    }


def verify_answer(
    answer: str,
    cited_line_ids: list[int],
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Check a draft answer against the lines actually retrieved, before saying it.

    Call this as the LAST step, always. It confirms every line you are about to cite
    was really returned by a retrieval tool in this conversation and really exists in
    the database.

    Args:
        answer: The answer you are about to give the user.
        cited_line_ids: Every line_id the answer relies on.

    Returns:
        `ok` plus, when it is False, the unsupported ids. If it is False, do not give
        the answer as drafted — drop the unsupported claims or retrieve again.
    """
    retrieved = set(tool_context.state.get(RETRIEVED_KEY, [])) if tool_context else set()
    cited = [int(i) for i in cited_line_ids]

    unsupported = sorted(set(cited) - retrieved)
    existing: set[int] = set()
    if cited:
        result = client().query(
            "SELECT line_id FROM lines WHERE line_id IN {ids:Array(UInt64)}",
            parameters={"ids": cited},
        )
        existing = {int(row[0]) for row in result.result_rows}
    nonexistent = sorted(set(cited) - existing)

    ok = not unsupported and not nonexistent and bool(cited or not answer.strip())
    return {
        "ok": ok,
        "cited": len(cited),
        "retrieved_in_session": len(retrieved),
        "not_retrieved_this_session": unsupported,
        "not_in_database": nonexistent,
        "verdict": (
            "grounded"
            if ok
            else "ungrounded - revise the answer to cite only verified lines, or say the data does not support it"
        ),
    }


def record_line_ids_from_rows(tool_context: ToolContext | None, payload: Any) -> int:
    """Harvest `line_id` values out of an arbitrary tool result and record them.

    Used for the MCP SQL tool, whose result shape this project does not control: it
    arrives as `{"columns": [...], "rows": [[...]]}`, sometimes JSON-encoded inside a
    text part. Anything unrecognisable is ignored rather than raising — a failure to
    harvest should cost a citation its support, never the whole answer.
    """
    if tool_context is None or payload is None:
        return 0

    def walk(node: Any) -> list[int]:
        if isinstance(node, str):
            try:
                return walk(json.loads(node))
            except (ValueError, TypeError):
                return []
        if isinstance(node, dict):
            columns = node.get("columns")
            rows = node.get("rows")
            if isinstance(columns, list) and isinstance(rows, list) and "line_id" in columns:
                index = columns.index("line_id")
                found = []
                for row in rows:
                    if isinstance(row, (list, tuple)) and len(row) > index:
                        try:
                            found.append(int(row[index]))
                        except (TypeError, ValueError):
                            continue
                return found
            return [i for value in node.values() for i in walk(value)]
        if isinstance(node, (list, tuple)):
            return [i for value in node for i in walk(value)]
        return []

    line_ids = walk(payload)
    _record(tool_context, line_ids)
    return len(line_ids)


# `run_sql` is deliberately NOT in this list: the ClickHouse track requires SQL to be
# executed through the official MCP server, so `build_agent` appends either the MCP
# toolset or `run_sql` as the fallback.
ALL_TOOLS = [
    inspect_schema,
    search_dialogue,
    aggregate_semantic_matches,
    verify_answer,
]
