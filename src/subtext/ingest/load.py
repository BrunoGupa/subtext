"""Load a corpus into ClickHouse, and build the chunk table for a given strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..chunking import Chunk, Line, chunk_lines, strategy_label
from ..db import client

LINE_COLUMNS: Sequence[str] = (
    "line_id", "title", "title_id", "season", "episode", "episode_title",
    "scene", "line_no", "character", "speaks_to", "timecode",
    "start_ms", "end_ms", "text",
)

CHUNK_COLUMNS: Sequence[str] = (
    "chunk_id", "strategy", "window_size", "line_id", "line_ids", "title_id",
    "season", "episode", "character", "start_ms", "text", "embedding",
)


def insert_lines(lines: Iterable[Line], *, replace_title_id: str | None = None) -> int:
    rows = [[getattr(ln, col) for col in LINE_COLUMNS] for ln in lines]
    c = client()
    if replace_title_id:
        c.command(
            "ALTER TABLE lines DELETE WHERE title_id = {tid:String}",
            parameters={"tid": replace_title_id},
        )
    if rows:
        c.insert("lines", rows, column_names=list(LINE_COLUMNS))
    return len(rows)


def read_lines() -> list[Line]:
    result = client().query(
        f"SELECT {', '.join(LINE_COLUMNS)} FROM lines ORDER BY title_id, season, episode, line_no"
    )
    return [Line(**dict(zip(LINE_COLUMNS, row))) for row in result.result_rows]


def insert_chunks(chunks: Iterable[Chunk], *, strategy: str, window_size: int) -> int:
    chunk_list = list(chunks)
    c = client()
    c.command(
        "ALTER TABLE line_chunks DELETE WHERE strategy = {s:String} AND window_size = {w:UInt8}",
        parameters={"s": strategy, "w": window_size},
    )
    if chunk_list:
        rows = [[getattr(ch, col) for col in CHUNK_COLUMNS] for ch in chunk_list]
        c.insert("line_chunks", rows, column_names=list(CHUNK_COLUMNS))
    return len(chunk_list)


def load_corpus(
    source: str = "sample",
    *,
    path: Path | None = None,
    title: str | None = None,
    title_id: str | None = None,
) -> int:
    """Load lines from `source` into the `lines` table. Returns the row count."""
    if source == "sample":
        from .sample import TITLE_ID, iter_lines

        return insert_lines(iter_lines(path), replace_title_id=TITLE_ID)

    if source == "srt":
        if path is None:
            raise ValueError("--path is required for --source srt")
        if not title or not title_id:
            raise ValueError("--title and --title-id are required for --source srt")
        from .srt import iter_lines

        return insert_lines(
            iter_lines(Path(path), title=title, title_id=title_id),
            replace_title_id=title_id,
        )

    raise ValueError(f"unknown source {source!r}; expected 'sample' or 'srt'")


def build_chunks(
    *,
    strategy: str = "line",
    window_size: int = 3,
    stride: int | None = None,
    embed_chunks: bool = True,
    show_progress: bool = False,
) -> int:
    """Chunk every loaded line with `strategy` and persist the chunks (embedded)."""
    from ..embeddings import embed

    lines = read_lines()
    chunks = list(chunk_lines(lines, strategy=strategy, window_size=window_size, stride=stride))
    if embed_chunks and chunks:
        vectors = embed([ch.text for ch in chunks], show_progress=show_progress)
        chunks = [
            Chunk(**{**ch.__dict__, "embedding": vector})
            for ch, vector in zip(chunks, vectors)
        ]
    effective_window = 1 if strategy == "line" else window_size
    insert_chunks(chunks, strategy=strategy, window_size=effective_window)
    return len(chunks)


def loaded_strategies() -> list[tuple[str, int, int]]:
    """`(strategy, window_size, chunk_count)` for everything currently in the table."""
    result = client().query(
        "SELECT strategy, window_size, count() AS n FROM line_chunks "
        "GROUP BY strategy, window_size ORDER BY strategy, window_size"
    )
    return [(row[0], int(row[1]), int(row[2])) for row in result.result_rows]


__all__ = [
    "load_corpus",
    "build_chunks",
    "read_lines",
    "loaded_strategies",
    "strategy_label",
]
