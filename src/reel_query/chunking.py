"""Chunking strategies.

A chunk is the unit that gets embedded and retrieved. Two strategies ship, and the
`evals` sweep varies them:

* ``line``   — one chunk per subtitle line. Precise, but a line like "I will. I promise."
               loses whatever it was promising.
* ``window`` — a sliding window of N consecutive lines from the same episode, carrying
               the surrounding dialogue. More context, blurrier attribution.

Both strategies keep ``line_ids``: the list of underlying lines a chunk covers. That is
what makes recall@k computable against a golden set labelled at *line* granularity, and
what lets a window-strategy hit be scored fairly against a line-strategy one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

STRATEGIES = ("line", "window")


@dataclass(frozen=True)
class Line:
    line_id: int
    title: str
    title_id: str
    season: int
    episode: int
    episode_title: str
    scene: int
    line_no: int
    character: str
    speaks_to: str
    timecode: str
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    strategy: str
    window_size: int
    line_id: int              # the anchor line (first line of the window)
    line_ids: list[int]       # every line the chunk covers
    title_id: str
    season: int
    episode: int
    character: str
    start_ms: int
    text: str
    embedding: list[float] = field(default_factory=list)


def _episode_key(line: Line) -> tuple[str, int, int]:
    return (line.title_id, line.season, line.episode)


def _render(lines: Sequence[Line]) -> str:
    return "\n".join(f"{ln.character}: {ln.text}" for ln in lines)


def chunk_lines(
    lines: Iterable[Line],
    *,
    strategy: str = "line",
    window_size: int = 3,
    stride: int | None = None,
) -> Iterator[Chunk]:
    """Yield chunks for the given strategy.

    ``window_size`` is ignored for ``line``. ``stride`` defaults to 1 (fully
    overlapping windows), which maximises recall at the cost of a larger table.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunk strategy {strategy!r}; expected one of {STRATEGIES}")

    ordered = sorted(lines, key=lambda ln: (ln.title_id, ln.season, ln.episode, ln.line_no))
    chunk_id = 0

    if strategy == "line":
        for ln in ordered:
            yield Chunk(
                chunk_id=chunk_id,
                strategy="line",
                window_size=1,
                line_id=ln.line_id,
                line_ids=[ln.line_id],
                title_id=ln.title_id,
                season=ln.season,
                episode=ln.episode,
                character=ln.character,
                start_ms=ln.start_ms,
                text=f"{ln.character}: {ln.text}",
            )
            chunk_id += 1
        return

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    step = stride if stride is not None else 1
    if step < 1:
        raise ValueError("stride must be >= 1")

    # Windows never cross an episode boundary — a promise made in S01E01 has nothing
    # to do with the line that happens to follow it in load order.
    bucket: list[Line] = []
    current: tuple[str, int, int] | None = None
    for ln in ordered + [None]:  # type: ignore[list-item]
        key = _episode_key(ln) if ln is not None else None
        if current is not None and key != current:
            for start in range(0, max(len(bucket) - window_size, 0) + 1, step):
                window = bucket[start : start + window_size]
                if not window:
                    continue
                anchor = window[0]
                yield Chunk(
                    chunk_id=chunk_id,
                    strategy="window",
                    window_size=window_size,
                    line_id=anchor.line_id,
                    line_ids=[w.line_id for w in window],
                    title_id=anchor.title_id,
                    season=anchor.season,
                    episode=anchor.episode,
                    # A window can span speakers; the anchor's character is the one
                    # the chunk is attributed to, and SQL can always re-join `lines`
                    # via line_ids when precise attribution matters.
                    character=anchor.character,
                    start_ms=anchor.start_ms,
                    text=_render(window),
                )
                chunk_id += 1
            bucket = []
        if ln is None:
            break
        bucket.append(ln)
        current = key


def strategy_label(strategy: str, window_size: int) -> str:
    return strategy if strategy == "line" else f"window{window_size}"
