"""Loader for real subtitles: a directory of `.srt` files (e.g. from OpenSubtitles).

Nothing about the corpus is bundled here — you point it at files you obtained
yourself. Season and episode are read from the filename (`S01E02`, `1x02`, ...), and
speaker attribution is read from the common `NAME:` / `- NAME:` subtitle convention
when it is present. Lines with no attributable speaker get `character = 'UNKNOWN'`
rather than being dropped, because they are still legitimate retrieval targets.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ..chunking import Line

_SEASON_EPISODE = (
    re.compile(r"[sS](\d{1,2})[\s._-]*[eE](\d{1,3})"),
    re.compile(r"(?<!\d)(\d{1,2})x(\d{1,3})(?!\d)"),
)
_TIMING = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")
_SPEAKER = re.compile(r"^\s*-?\s*([A-Z][A-Z '’.\-]{1,30}):\s*(.+)$")
_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def parse_season_episode(name: str) -> tuple[int, int]:
    for pattern in _SEASON_EPISODE:
        match = pattern.search(name)
        if match:
            return int(match.group(1)), int(match.group(2))
    return 0, 0


def _to_ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def _clean(text: str) -> str:
    text = _TAGS.sub("", text)
    text = _BRACKETED.sub("", text)
    return " ".join(text.split())


def parse_srt(path: Path) -> Iterator[tuple[int, int, str]]:
    """Yield `(start_ms, end_ms, text)` for each cue in an .srt file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    for block in re.split(r"\r?\n\r?\n", raw):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        timing = next((_TIMING.search(ln) for ln in lines[:2] if _TIMING.search(ln)), None)
        if timing is None:
            continue
        start = _to_ms(*timing.group(1, 2, 3, 4))
        end = _to_ms(*timing.group(5, 6, 7, 8))
        body_index = lines.index(timing.string if timing.string in lines else lines[1]) + 1
        text = " ".join(lines[body_index:]) if body_index < len(lines) else ""
        text = _clean(text)
        if text:
            yield start, end, text


def iter_lines(
    directory: Path,
    *,
    title: str,
    title_id: str,
    start_line_id: int = 1,
) -> Iterator[Line]:
    from .sample import format_timecode

    line_id = start_line_id
    for path in sorted(directory.glob("**/*.srt")):
        season, episode = parse_season_episode(path.stem)
        line_no = 0
        for start_ms, end_ms, text in parse_srt(path):
            speaker_match = _SPEAKER.match(text)
            if speaker_match:
                character = speaker_match.group(1).strip().title()
                text = speaker_match.group(2).strip()
            else:
                character = "UNKNOWN"
            if not text:
                continue
            line_no += 1
            yield Line(
                line_id=line_id,
                title=title,
                title_id=title_id,
                season=season,
                episode=episode,
                episode_title=path.stem,
                scene=0,
                line_no=line_no,
                character=character,
                speaks_to="",
                timecode=format_timecode(start_ms),
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
            line_id += 1
