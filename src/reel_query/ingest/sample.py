"""Loader for the bundled synthetic corpus in `data/sample/`.

Timecodes are synthesised: the corpus is authored as ordered dialogue without
timings, and the pipeline needs `start_ms` / `end_ms` to be real columns because the
demo question filters by season *and* timecode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..chunking import Line
from ..config import DATA_DIR

SAMPLE_DIR = DATA_DIR / "sample"
TITLE = "The Lighthouse Contract"
TITLE_ID = "tt_lighthouse"

LINE_MS = 3_400          # nominal duration of one spoken line
GAP_MS = 700             # nominal pause between lines
SCENE_GAP_MS = 9_000     # extra pause on a scene change


def format_timecode(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def iter_lines(sample_dir: Path | None = None) -> Iterator[Line]:
    directory = sample_dir or SAMPLE_DIR
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl files in {directory}")

    line_id = 0
    for path in files:
        cursor = 0
        previous_episode: tuple[int, int] | None = None
        previous_scene: int | None = None
        line_no = 0

        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            episode_key = (record["season"], record["episode"])
            if episode_key != previous_episode:
                cursor = 0
                line_no = 0
                previous_scene = None
                previous_episode = episode_key
            if previous_scene is not None and record["scene"] != previous_scene:
                cursor += SCENE_GAP_MS
            previous_scene = record["scene"]

            start_ms = cursor
            end_ms = start_ms + LINE_MS
            cursor = end_ms + GAP_MS
            line_no += 1
            line_id += 1

            yield Line(
                line_id=line_id,
                title=TITLE,
                title_id=TITLE_ID,
                season=record["season"],
                episode=record["episode"],
                episode_title=record["episode_title"],
                scene=record["scene"],
                line_no=line_no,
                character=record["character"],
                speaks_to=record.get("speaks_to", ""),
                timecode=format_timecode(start_ms),
                start_ms=start_ms,
                end_ms=end_ms,
                text=record["text"],
            )
