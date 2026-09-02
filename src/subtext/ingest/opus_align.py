"""Recover film metadata for a Moses bitext from the OPUS sentence-alignment file.

The Moses distribution is two bare text files: no title, no year, no film id. All of
that lives in the *alignment* file (`<pair>.xml.gz`), whose ``linkGrp`` elements name
the two documents they align:

    <linkGrp fromDoc="en/1954/0047478/1954690845.xml.gz"
             toDoc="es/1954/0047478/1954223667.xml.gz">
      <link id="SL0" xtargets="1;1" />
      <link id="SL1" xtargets="2 3;2" />
      <link id="SL2" xtargets=";4" />     <- Spanish only: NOT a Moses line
      ...

The path carries the release year and the IMDb id (zero-padded, no ``tt`` prefix).
Each ``link`` with text on *both* sides becomes exactly one line in each Moses file,
in order, so counting those links per document reconstructs which line range belongs
to which film.

Links with an empty side are dropped by the Moses build, which is why the counting
has to be selective — this is the part that silently goes wrong if you assume one
link is one line.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_LINK_GRP = re.compile(rb'fromDoc="([^"]+)"')
_XTARGETS = re.compile(rb'xtargets="([^"]*)"')
_DOC_PATH = re.compile(r"^[a-z]{2,3}/(\d{1,4})/([^/]+)/")


@dataclass(frozen=True)
class Document:
    """One aligned film or episode, and the Moses line range it occupies."""

    ordinal: int
    start_line: int          # 0-based, inclusive
    n_lines: int
    year: int                # 0 when the path carries a nonsense year
    imdb_id: str             # the film, or the *episode*, as tt-prefixed
    series_id: str = ""      # the parent series, episodes only
    season: int = 0
    episode: int = 0

    @property
    def kind(self) -> str:
        return "episode" if self.series_id else "movie"

    @property
    def end_line(self) -> int:
        return self.start_line + self.n_lines


def parse_doc_path(path: str) -> tuple[int, str, str, int, int]:
    """`'en/1954/0047478/123.xml.gz'` -> `(1954, 'tt0047478', '', 0, 0)`.

    The folder is one of exactly two shapes across the whole corpus:

    * ``0047478`` — a film, the plain IMDb number.
    * ``3276470_2741602_1_19`` — an episode: *episode* IMDb id, *series* IMDb id,
      season, episode. Both halves are real ids, so a TV line can be resolved to
      its episode title *and* its series.

    A handful of paths carry an impossible year (``1191``); those come back as 0
    rather than raising, since dropping the document would bias the sample toward
    well-formed metadata.
    """
    match = _DOC_PATH.match(path)
    if not match:
        return 0, "", "", 0, 0
    raw_year, folder = match.group(1), match.group(2)
    year = int(raw_year)
    if not (1890 <= year <= 2100):
        year = 0

    if folder.isdigit():
        return year, f"tt{int(folder):07d}", "", 0, 0

    parts = folder.split("_")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        episode_id, series_id, season, episode = parts
        return (
            year,
            f"tt{int(episode_id):07d}",
            f"tt{int(series_id):07d}",
            int(season),
            int(episode),
        )
    return year, "", "", 0, 0


def iter_documents(alignment_path: Path) -> Iterator[Document]:
    """Stream the alignment file, yielding one `Document` per aligned film."""
    ordinal = 0
    cursor = 0
    current_path: str | None = None
    kept = 0

    with gzip.open(alignment_path, "rb") as stream:
        for raw in stream:
            group = _LINK_GRP.search(raw)
            if group is not None:
                if current_path is not None:
                    year, imdb_id, series_id, season, episode = parse_doc_path(current_path)
                    yield Document(
                        ordinal, cursor, kept, year, imdb_id, series_id, season, episode
                    )
                    ordinal += 1
                    cursor += kept
                current_path = group.group(1).decode("utf-8", "replace")
                kept = 0
                continue

            targets = _XTARGETS.search(raw)
            if targets is not None:
                left, _, right = targets.group(1).partition(b";")
                if left.strip() and right.strip():
                    kept += 1

    if current_path is not None:
        year, imdb_id, series_id, season, episode = parse_doc_path(current_path)
        yield Document(ordinal, cursor, kept, year, imdb_id, series_id, season, episode)


def write_index(alignment_path: Path, out_path: Path) -> tuple[int, int]:
    """Write a TSV index of documents. Returns `(documents, total_lines)`."""
    documents = 0
    total_lines = 0
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "ordinal\tstart_line\tn_lines\tyear\timdb_id\tseries_id\tseason\tepisode\n"
        )
        for doc in iter_documents(alignment_path):
            handle.write(
                f"{doc.ordinal}\t{doc.start_line}\t{doc.n_lines}\t{doc.year}\t"
                f"{doc.imdb_id}\t{doc.series_id}\t{doc.season}\t{doc.episode}\n"
            )
            documents += 1
            total_lines += doc.n_lines
    return documents, total_lines


__all__ = ["Document", "parse_doc_path", "iter_documents", "write_index"]
