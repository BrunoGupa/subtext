"""Local sentence-transformer embeddings. No API calls, no cost."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .config import settings


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: Sequence[str], *, batch_size: int = 256, show_progress: bool = False) -> list[list[float]]:
    """Embed a batch of texts into unit-normalised vectors."""
    if not texts:
        return []
    model = _model(settings().embedding_model)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [[float(x) for x in row] for row in vectors]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def dimension() -> int:
    return int(_model(settings().embedding_model).get_sentence_embedding_dimension())


def embed_column(
    *,
    source: str,
    column: str,
    target: str,
    batch_size: int = 10_000,
    min_length: int = 1,
    log=print,
) -> dict:
    """Embed every distinct value of `column` in `source` into `target`.

    Distinct, because a subtitle corpus repeats itself heavily — "What?" occurs 2,140
    times in `mx_corpus` and one vector answers for all of them. Rows stream through in
    batches so peak memory stays flat regardless of corpus size.

    Runs on the CPU with a local sentence-transformer. No API, no key, no cost.
    """
    import time

    from . import db

    ch = db.client()
    ch.command(f"DROP TABLE IF EXISTS {target}")
    ch.command(
        f"CREATE TABLE {target} (text String, embedding Array(Float32)) "
        f"ENGINE = MergeTree ORDER BY text"
    )

    total = ch.query(
        f"SELECT uniqExact({column}) FROM {source} WHERE length({column}) >= {min_length}"
    ).result_rows[0][0]
    log(f"  {total:,} distinct values to embed")

    started = time.perf_counter()
    done = 0
    while True:
        rows = ch.query(
            f"SELECT DISTINCT {column} AS t FROM {source} "
            f"WHERE length({column}) >= {min_length} "
            f"ORDER BY t LIMIT {batch_size} OFFSET {done}"
        ).result_rows
        if not rows:
            break
        texts = [r[0] for r in rows]
        vectors = embed(texts, batch_size=256)
        ch.insert(target, [[t, v] for t, v in zip(texts, vectors)],
                  column_names=["text", "embedding"])
        done += len(texts)
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0
        log(f"  {done:,}/{total:,}  {rate:,.0f}/s  eta {(total-done)/rate/60:.1f} min"
            if rate else f"  {done:,}/{total:,}")

    return {
        "vectors": done,
        "dimension": dimension(),
        "seconds": round(time.perf_counter() - started, 1),
    }
