"""Download the third-party corpora this project analyses.

**No corpus is redistributed with this repository.** The datasets below are large,
carry their own licences, and in the case of OpenSubtitles consist of text OPUS does
not own and will remove on request. Shipping them from here would be both rude and
legally careless. So the *loader* is the deliverable and the data is fetched from its
original host, the same arrangement `torchvision.datasets(download=True)` and the
HuggingFace dataset scripts use.

Anything downloaded lands in `data/raw/`, which is gitignored.

Licence obligations you inherit by running this
------------------------------------------------
**OpenSubtitles / OPUS** — the README shipped inside the archive is explicit:

    Please, add a link to http://www.opensubtitles.org/ to your website and to your
    reports and publications produced with the data!

and asks that you cite:

    P. Lison and J. Tiedemann, 2016. *OpenSubtitles2016: Extracting Large Parallel
    Corpora from Movie and TV Subtitles.* LREC 2016.

**IMDb** — the bulk datasets are free for *personal and non-commercial* use, and
require attribution to IMDb. See https://developer.imdb.com/non-commercial-datasets/

Both obligations are restated in `CORPUS.md`. Honour them.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import DATA_DIR

RAW_DIR = DATA_DIR / "raw"


@dataclass(frozen=True)
class Dataset:
    key: str
    url: str
    destination: Path
    approx_bytes: int
    description: str
    sha256: str = ""          # '' while unpinned; a known hash is verified after download

    @property
    def approx_size(self) -> str:
        return human_size(self.approx_bytes)


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


OPUS_BASE = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024"
IMDB_BASE = "https://datasets.imdbws.com"


def opus_datasets(lang_pair: str = "en-es") -> list[Dataset]:
    """The two OPUS files: the bitext itself, and the alignment file that carries
    year / IMDb id for every aligned document."""
    return [
        Dataset(
            key=f"opus-{lang_pair}-moses",
            url=f"{OPUS_BASE}/moses/{lang_pair}.txt.zip",
            destination=RAW_DIR / "opus" / f"{lang_pair}.txt.zip",
            approx_bytes=2_676_648_584,
            description=f"OpenSubtitles v2024 {lang_pair} sentence pairs (Moses format)",
            sha256=(
                "be8dcb734b196340f0364fce8b151d5930aed52dc3c57c86f3a8da9fb4111bd0"
                if lang_pair == "en-es" else ""
            ),
        ),
        Dataset(
            key=f"opus-{lang_pair}-align",
            url=f"{OPUS_BASE}/xml/{lang_pair}.xml.gz",
            destination=RAW_DIR / "opus" / f"{lang_pair}.xml.gz",
            approx_bytes=1_095_445_521,
            description=f"OpenSubtitles v2024 {lang_pair} sentence alignment (film year + IMDb id)",
            sha256=(
                "6e6d259d174cfd4db66c7b880312885da1b0643cec4b006830740e079368ea25"
                if lang_pair == "en-es" else ""
            ),
        ),
    ]


# IMDb regenerates these files every day, so a pinned hash would fail for everyone
# tomorrow. OPUS v2024 is a frozen release and *is* pinned above. Verify IMDb by size
# and by the row counts `subtext status` reports, not by checksum.
IMDB_DATASETS: list[Dataset] = [
    Dataset(
        key="imdb-basics",
        url=f"{IMDB_BASE}/title.basics.tsv.gz",
        destination=RAW_DIR / "imdb" / "title.basics.tsv.gz",
        approx_bytes=226_249_654,
        description="IMDb titles: primary/original title, type, year, genres",
    ),
    Dataset(
        key="imdb-akas",
        url=f"{IMDB_BASE}/title.akas.tsv.gz",
        destination=RAW_DIR / "imdb" / "title.akas.tsv.gz",
        approx_bytes=512_390_718,
        description="IMDb regional titles: region (MX, ES, US...), language",
    ),
]


def all_datasets(lang_pair: str = "en-es") -> dict[str, Dataset]:
    every = opus_datasets(lang_pair) + IMDB_DATASETS
    return {d.key: d for d in every}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    dataset: Dataset,
    *,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Fetch one dataset. Existing files are kept unless `force`.

    Downloads to a `.part` file and renames on success, so an interrupted run never
    leaves a truncated archive that looks complete.
    """
    destination = dataset.destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(dataset.url, headers={"User-Agent": "subtext/0.1"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            seen += len(chunk)
            if on_progress is not None:
                on_progress(seen, total)

    if dataset.sha256:
        actual = sha256_of(partial)
        if actual != dataset.sha256:
            partial.unlink(missing_ok=True)
            raise ValueError(
                f"{dataset.key}: checksum mismatch\n  expected {dataset.sha256}\n  got      {actual}"
            )

    shutil.move(str(partial), str(destination))
    return destination


__all__ = [
    "Dataset",
    "RAW_DIR",
    "all_datasets",
    "opus_datasets",
    "IMDB_DATASETS",
    "download",
    "sha256_of",
    "human_size",
]
