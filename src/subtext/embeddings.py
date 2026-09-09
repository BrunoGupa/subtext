"""Embeddings, from Google's `gemini-embedding-001`.

They were local sentence-transformers until 2026-09-08. That is a Microsoft-lineage model
and the hackathon permits only Google Cloud AI tooling, so the swap was an eligibility
requirement -- but it also measured better on the case this project keeps failing:
`He wore this watch up his ass` against 700 corpus distractors found 0 of 6 `up his ass`
lines with MiniLM and 5 of 5 here.

Three decisions, and the first two are baked into every stored vector:

* **`SEMANTIC_SIMILARITY`, not `RETRIEVAL_*`.** Google's own split is symmetry: retrieval
  task types give queries and documents *different* representations, while semantic
  similarity treats both sides alike and is documented for "duplicate detection". A
  subtitle line looked up against subtitle lines is the symmetric case, and measurement
  agreed -- 15 hits against 14, and 5/5 against 4/5 on the hard cue.
* **768 dimensions, not 3072.** Measured identical on retrieval (4/5 either way at the
  time of the sweep) while storing a quarter as much: 829 MB against 3.3 GB for 269,869
  vectors, on a cluster sized at 8 GiB.
* **Normalised here, by hand.** Google is explicit that for `gemini-embedding-001` users
  "must manually normalize non-3072 dimensions". Skipping it does not raise: it silently
  distorts every cosine the project computes.
"""

from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Sequence

from .config import settings

#: What the vectors are for. See the note above -- this is stored, not a query-time knob.
TASK_TYPE = "SEMANTIC_SIMILARITY"

#: Texts per request. The API accepts 100; it also silently returns fewer embeddings than
#: it was given, which is why `embed` checks rather than trusts.
BATCH = 100

#: Requests in flight. The quota is 3,000 requests a minute and one request takes about
#: two seconds, so this is nowhere near it -- it is sized against latency, not the limit.
CONCURRENCY = 8


@lru_cache(maxsize=1)
def _client():
    from google import genai

    return genai.Client()


def embed(texts: Sequence[str], *, batch_size: int = BATCH, show_progress: bool = False,
          attempts: int = 8) -> list[list[float]]:
    """Embed texts into unit-normalised vectors, in order, one per input.

    Two failure modes are handled because both are silent. The API can return fewer
    embeddings than it was asked for -- `gemini-embedding-2` returns one for any batch --
    and a short batch accepted without checking would shift every later vector onto the
    wrong line. And a transient error mid-corpus would otherwise abandon a paid run.
    """
    from google.genai import types

    if not texts:
        return []

    model = settings().embedding_model
    dim = settings().embedding_dim
    config = types.EmbedContentConfig(task_type=TASK_TYPE, output_dimensionality=dim)

    # Chunks are sent concurrently and reassembled by position. The bound is latency, not
    # quota: one 100-text request takes about two seconds, so sending them one at a time
    # ran at ~36 texts/s while sitting at 30 requests per minute against a limit of 3,000.
    # Order is restored by index rather than by completion, because a vector that lands
    # against the wrong line is a silent corruption of the whole corpus.
    chunks = [list(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
    results: list[list[list[float]]] = [[] for _ in chunks]

    def run(idx: int) -> None:
        chunk, size = chunks[idx], len(chunks[idx])
        while True:
            got = _embed_chunk(chunk[:size], model, config, attempts)
            if len(got) == len(chunk[:size]):
                break
            if size == 1:
                raise RuntimeError(
                    f"{model} returned {len(got)} embeddings for a single text")
            size = max(1, size // 4)
        # A short batch is retried whole at a smaller size rather than stitched, so the
        # chunk either arrives complete or not at all.
        if size < len(chunk):
            results[idx] = embed(chunk, batch_size=size, attempts=attempts)
        else:
            results[idx] = [_normalise(v) for v in got]

    if len(chunks) == 1:
        run(0)
    else:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            for future in as_completed([pool.submit(run, i) for i in range(len(chunks))]):
                future.result()

    out = [v for chunk in results for v in chunk]
    assert len(out) == len(texts), f"{len(out)} vectors for {len(texts)} texts"
    if show_progress:
        print(f"    {len(out):,}", end="\r")
    return out


def _embed_chunk(chunk, model, config, attempts) -> list[list[float]]:
    for attempt in range(attempts):
        try:
            reply = _client().models.embed_content(
                model=model, contents=chunk, config=config)
            return [list(e.values) for e in reply.embeddings]
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            # A 429 carries the wait the server wants and it is measured in the
            # minute-long quota window, so exponential backoff from one second gives up
            # while the window is still closed: 1+2+4+8 seconds abandoned a paid run
            # 5,000 rows in. Honour the server's own number.
            time.sleep(_retry_delay(exc) or min(60.0, 5.0 * (attempt + 1)))
    return []


def _retry_delay(exc: Exception) -> float:
    """The `retryDelay` a quota error asks for, in seconds, or 0.0 if it names none."""
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(exc))
    return float(match.group(1)) + 1.0 if match else 0.0


def _normalise(vector: list[float]) -> list[float]:
    """Unit length. Required below 3072 dimensions, and harmless at 3072."""
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def dimension() -> int:
    return settings().embedding_dim


#: One HNSW graph should span the whole part. See the trap described in `embed_column`.
VECTOR_INDEX_GRANULARITY = 100_000_000


def embed_column(
    *,
    source: str,
    column: str,
    target: str,
    batch_size: int = 10_000,
    min_length: int = 1,
    index: bool = True,
    resume: bool = False,
    log=print,
) -> dict:
    """Embed every distinct value of `column` in `source` into `target`.

    Distinct, because a subtitle corpus repeats itself heavily — "What?" occurs 2,140
    times in `mx_corpus` and one vector answers for all of them. Rows stream through in
    batches so peak memory stays flat regardless of corpus size.

    Embedding is a paid API call. Each batch is inserted before the next is requested, so
    a crash leaves the work done so far in the table -- but this function DROPs the target
    first, so re-running it starts over and pays again. Resuming a half-finished corpus
    means dropping that line and paging from the existing row count.

    An HNSW index is built afterwards unless `index=False`. Note the granularity: a
    vector index wants ONE graph spanning the whole part, so GRANULARITY must be large.
    Left at the default of 1 each graph covers a single 8,192-row granule, and the search
    returns confident nonsense -- "Where is my car?" came back with "- About gay stuff."
    That is a configuration trap, not a ClickHouse defect, and it is silent: the query
    succeeds and the rows look plausible until you compare them against a full scan.
    """
    import time

    from . import db

    ch = db.client()
    if resume:
        # Vectors already bought are already in the table. `ORDER BY t` makes the paging
        # deterministic, so the row count is a valid offset into the same sequence.
        done_already = ch.query(f"SELECT count() FROM {target}").result_rows[0][0]
        log(f"  resuming: {done_already:,} already embedded")
    else:
        done_already = 0
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
    done = done_already
    while True:
        rows = ch.query(
            f"SELECT DISTINCT {column} AS t FROM {source} "
            f"WHERE length({column}) >= {min_length} "
            f"ORDER BY t LIMIT {batch_size} OFFSET {done}"
        ).result_rows
        if not rows:
            break
        texts = [r[0] for r in rows]
        vectors = embed(texts)
        ch.insert(target, [[t, v] for t, v in zip(texts, vectors)],
                  column_names=["text", "embedding"])
        done += len(texts)
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0
        log(f"  {done:,}/{total:,}  {rate:,.0f}/s  eta {(total-done)/rate/60:.1f} min"
            if rate else f"  {done:,}/{total:,}")

    stats = {
        "vectors": done,
        "dimension": dimension(),
        "seconds": round(time.perf_counter() - started, 1),
    }

    if index and done:
        t = time.perf_counter()
        log("  building the HNSW index ...")
        settings = {"allow_experimental_vector_similarity_index": 1, "mutations_sync": 2}
        ch.command(
            f"ALTER TABLE {target} ADD INDEX idx_vec embedding "
            f"TYPE vector_similarity('hnsw', 'cosineDistance', {dimension()}) "
            f"GRANULARITY {VECTOR_INDEX_GRANULARITY}",
            settings=settings,
        )
        ch.command(f"ALTER TABLE {target} MATERIALIZE INDEX idx_vec", settings=settings)
        stats["index_seconds"] = round(time.perf_counter() - t, 1)

    return stats
