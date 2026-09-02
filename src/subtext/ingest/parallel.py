"""Loader for sentence-aligned parallel corpora in Moses format.

Moses format is two plain-text files with one sentence per line, aligned by line
number: line *n* of ``corpus.en-es.en`` is the translation of line *n* of
``corpus.en-es.es``. That is the whole contract, and it is what makes the
translation questions answerable — a rendering is just the other side of a row.

Nothing is bundled with this repo. Point it at an archive you obtained yourself
(OPUS distributes OpenSubtitles this way); `data/raw/` is gitignored.

The archive is read as a stream, so a 2.7 GB zip never becomes 10 GB of text on
disk.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterator, NamedTuple

MAX_CHARS = 400          # subtitle cues are short; anything longer is a merge artefact
MIN_CHARS = 2


class Pair(NamedTuple):
    source_text: str
    target_text: str


def _member_names(archive: zipfile.ZipFile, lang_pair: str) -> tuple[str, str]:
    """Locate the two aligned text members for `lang_pair` (e.g. 'en-es')."""
    source_lang, target_lang = lang_pair.split("-")
    members = [n for n in archive.namelist() if not n.endswith("/")]

    def find(lang: str) -> str:
        matches = [n for n in members if n.endswith(f".{lang_pair}.{lang}")]
        if not matches:
            raise FileNotFoundError(
                f"no member ending in '.{lang_pair}.{lang}' in the archive; "
                f"found: {', '.join(sorted(members)[:8])}"
            )
        return sorted(matches)[0]

    return find(source_lang), find(target_lang)


def iter_pairs(
    archive_path: Path,
    *,
    lang_pair: str = "en-es",
    limit: int | None = None,
    skip: int = 0,
    stride: int = 1,
) -> Iterator[Pair]:
    """Yield aligned `(source, target)` pairs, streamed from the zip.

    Rows where either side is empty, absurdly long, or identical on both sides
    (untranslated boilerplate — song titles, names, numerals) are dropped: they are
    noise for every question this corpus exists to answer.

    `stride` keeps every Nth pair. The corpus is ordered by film id, which correlates
    with release date, so a contiguous head-of-file slice is a sample of *old* cinema
    and misses register that only appears in modern releases. Striding spreads the
    sample across the whole collection instead.
    """
    with zipfile.ZipFile(archive_path) as archive:
        source_name, target_name = _member_names(archive, lang_pair)
        with archive.open(source_name) as raw_source, archive.open(target_name) as raw_target:
            source_stream = io.TextIOWrapper(raw_source, encoding="utf-8", errors="replace")
            target_stream = io.TextIOWrapper(raw_target, encoding="utf-8", errors="replace")

            emitted = 0
            for index, (source_line, target_line) in enumerate(zip(source_stream, target_stream)):
                if index < skip:
                    continue
                if stride > 1 and (index - skip) % stride:
                    continue
                source_text = source_line.strip()
                target_text = target_line.strip()
                if not (MIN_CHARS <= len(source_text) <= MAX_CHARS):
                    continue
                if not (MIN_CHARS <= len(target_text) <= MAX_CHARS):
                    continue
                if source_text == target_text:
                    continue
                yield Pair(source_text, target_text)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def load_parallel(
    archive_path: Path,
    *,
    lang_pair: str = "en-es",
    corpus: str = "opensubtitles",
    limit: int | None = None,
    skip: int = 0,
    stride: int = 1,
    batch_size: int = 100_000,
    replace: bool = True,
) -> int:
    """Stream a Moses archive into `aligned_lines`. Returns the row count inserted."""
    from ..db import client

    source_lang, target_lang = lang_pair.split("-")
    c = client()
    if replace:
        c.command(
            "ALTER TABLE aligned_lines DELETE WHERE corpus = {c:String} "
            "AND lang_pair = {lp:String} SETTINGS mutations_sync = 1",
            parameters={"c": corpus, "lp": lang_pair},
        )

    columns = [
        "pair_id", "corpus", "lang_pair", "source_lang", "target_lang",
        "source_text", "target_text",
    ]
    batch: list[list[object]] = []
    total = 0

    for pair_id, pair in enumerate(
        iter_pairs(
            archive_path, lang_pair=lang_pair, limit=limit, skip=skip, stride=stride
        ),
        start=1,
    ):
        batch.append([
            pair_id, corpus, lang_pair, source_lang, target_lang,
            pair.source_text, pair.target_text,
        ])
        if len(batch) >= batch_size:
            c.insert("aligned_lines", batch, column_names=columns)
            total += len(batch)
            batch = []
    if batch:
        c.insert("aligned_lines", batch, column_names=columns)
        total += len(batch)
    return total


__all__ = ["Pair", "iter_pairs", "load_parallel"]
