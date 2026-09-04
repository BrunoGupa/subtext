"""Retrieval over ClickHouse.

Two entry points, and the difference between them is the whole architecture:

* :func:`search` — plain vector similarity. Ranked dialogue, nothing aggregated.
* :func:`hybrid_aggregate` — **one** SQL statement in which the vector search is a CTE
  that *feeds* a GROUP BY over the structured columns. This is the path the demo
  question takes: "break a promise" is resolved semantically, and "how many times,
  per season, for this character" is resolved by SQL over the rows the vector search
  returned. Neither half can answer it alone, and they are not two features glued
  together at the application layer — it is a single query plan inside ClickHouse.

Similarity is brute-force ``cosineDistance`` over ``Array(Float32)``. At this corpus
size that is fast, and it avoids depending on the experimental HNSW index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .db import client
from .embeddings import embed_one


@dataclass(frozen=True)
class Hit:
    line_id: int
    chunk_id: int
    line_ids: list[int]
    title_id: str
    season: int
    episode: int
    character: str
    timecode: str
    text: str
    distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


def _filter_clause(
    *,
    title_id: str | None,
    character: str | None,
    season: int | None,
    episode: int | None,
    prefix: str,
    involving: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the structural WHERE clause.

    `character` and `involving` are not the same filter, and confusing them quietly
    breaks the headline question. "How many times does Vale break a promise?" is
    almost never evidenced by a line Vale *speaks* — it is evidenced by Marisol saying
    *"You said an hour"*. `character` scopes to the speaker; `involving` scopes to the
    conversation, speaker or addressee.
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if title_id:
        clauses.append(f"{prefix}title_id = {{f_title:String}}")
        params["f_title"] = title_id
    if character:
        clauses.append(f"lower({prefix}character) = lower({{f_char:String}})")
        params["f_char"] = character
    if involving:
        clauses.append(
            f"(lower({prefix}character) = lower({{f_inv:String}})"
            f" OR lower({prefix}speaks_to) = lower({{f_inv:String}}))"
        )
        params["f_inv"] = involving
    if season is not None:
        clauses.append(f"{prefix}season = {{f_season:UInt16}}")
        params["f_season"] = season
    if episode is not None:
        clauses.append(f"{prefix}episode = {{f_episode:UInt16}}")
        params["f_episode"] = episode
    return (" AND ".join(clauses) if clauses else "1"), params


# The vector-search CTE. `over_fetch` widens the candidate pool *before* the
# structural filters are applied, so filtering by character does not silently
# shrink an already-truncated top-k down to nothing.
_CANDIDATES_CTE = """
WITH candidates AS (
    SELECT
        chunk_id,
        line_ids,
        title_id,
        cosineDistance(embedding, {q:Array(Float32)}) AS distance
    FROM line_chunks
    WHERE strategy = {strategy:String}
      AND window_size = {window_size:UInt8}
      AND length(embedding) > 0
    ORDER BY distance ASC
    LIMIT {pool:UInt32}
)
"""


def search(
    question: str,
    *,
    k: int = 10,
    strategy: str = "line",
    window_size: int = 1,
    title_id: str | None = None,
    character: str | None = None,
    involving: str | None = None,
    season: int | None = None,
    episode: int | None = None,
    max_distance: float | None = None,
    over_fetch: int = 8,
) -> list[Hit]:
    """Vector search, resolved back to individual dialogue lines."""
    where, params = _filter_clause(
        title_id=title_id, character=character, season=season, episode=episode,
        prefix="l.", involving=involving,
    )
    distance_clause = "AND distance <= {max_distance:Float32}" if max_distance is not None else ""
    sql = f"""
    {_CANDIDATES_CTE}
    SELECT
        l.line_id AS line_id, c.chunk_id AS chunk_id, c.line_ids AS line_ids,
        l.title_id AS title_id, l.season AS season, l.episode AS episode,
        l.character AS character, l.timecode AS timecode, l.text AS text,
        c.distance AS distance
    FROM candidates AS c
    ARRAY JOIN c.line_ids AS lid
    INNER JOIN lines AS l ON l.line_id = lid AND l.title_id = c.title_id
    WHERE {where} {distance_clause}
    ORDER BY c.distance ASC, l.line_id ASC
    LIMIT {{k:UInt32}}
    """
    params |= {
        "q": embed_one(question),
        "strategy": strategy,
        "window_size": window_size,
        "pool": max(k * over_fetch, k),
        "k": k,
    }
    if max_distance is not None:
        params["max_distance"] = max_distance

    result = client().query(sql, parameters=params)
    seen: set[int] = set()
    hits: list[Hit] = []
    for row in result.result_rows:
        record = dict(zip(result.column_names, row))
        if record["line_id"] in seen:
            continue
        seen.add(record["line_id"])
        hits.append(
            Hit(
                line_id=int(record["line_id"]),
                chunk_id=int(record["chunk_id"]),
                line_ids=[int(x) for x in record["line_ids"]],
                title_id=record["title_id"],
                season=int(record["season"]),
                episode=int(record["episode"]),
                character=record["character"],
                timecode=record["timecode"],
                text=record["text"],
                distance=float(record["distance"]),
            )
        )
    return hits


ALLOWED_GROUP_BY = {
    "season": "l.season",
    "episode": "l.episode",
    "character": "l.character",
    "title_id": "l.title_id",
    "episode_title": "l.episode_title",
}


def hybrid_aggregate(
    question: str,
    *,
    group_by: Sequence[str] = ("season",),
    k: int = 40,
    strategy: str = "line",
    window_size: int = 1,
    title_id: str | None = None,
    character: str | None = None,
    involving: str | None = None,
    season: int | None = None,
    episode: int | None = None,
    max_distance: float | None = 0.65,
    over_fetch: int = 8,
) -> tuple[list[dict[str, Any]], str]:
    """Semantic candidates -> SQL GROUP BY, in a single ClickHouse query.

    Returns ``(rows, sql)``. The SQL comes back so the agent can show its work and so
    the answer can be audited against the exact query that produced it.
    """
    unknown = [g for g in group_by if g not in ALLOWED_GROUP_BY]
    if unknown:
        raise ValueError(f"cannot group by {unknown}; allowed: {sorted(ALLOWED_GROUP_BY)}")

    where, params = _filter_clause(
        title_id=title_id, character=character, season=season, episode=episode,
        prefix="l.", involving=involving,
    )
    distance_clause = "AND distance <= {max_distance:Float32}" if max_distance is not None else ""
    group_expressions = [ALLOWED_GROUP_BY[g] for g in group_by]
    select_list = ", ".join(f"{expr} AS {name}" for name, expr in zip(group_by, group_expressions))
    group_list = ", ".join(group_expressions)

    sql = f"""
    {_CANDIDATES_CTE}
    SELECT
        {select_list},
        count(DISTINCT l.line_id) AS matches,
        round(min(c.distance), 4)  AS best_distance,
        groupArray(8)(l.line_id)   AS line_ids,
        groupArray(4)(l.text)      AS examples
    FROM candidates AS c
    ARRAY JOIN c.line_ids AS lid
    INNER JOIN lines AS l ON l.line_id = lid AND l.title_id = c.title_id
    WHERE {where} {distance_clause}
    GROUP BY {group_list}
    ORDER BY matches DESC, {group_list}
    """
    params |= {
        "q": embed_one(question),
        "strategy": strategy,
        "window_size": window_size,
        "pool": max(k * over_fetch, k),
    }
    if max_distance is not None:
        params["max_distance"] = max_distance

    result = client().query(sql, parameters=params)
    rows = [dict(zip(result.column_names, row)) for row in result.result_rows]
    return rows, " ".join(sql.split())


def corpus_stats() -> dict[str, Any]:
    result = client().query(
        """
        SELECT
            count() AS lines,
            uniqExact(title_id) AS titles,
            uniqExact(character) AS characters,
            max(season) AS seasons
        FROM lines
        """
    )
    if not result.result_rows:
        return {"lines": 0, "titles": 0, "characters": 0, "seasons": 0}
    return dict(zip(result.column_names, result.result_rows[0]))


# --------------------------------------------------------------------------------------
# Localisation retrieval: find Mexican precedent for an English cue
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Precedent:
    """One attested rendering, with everything needed to verify it."""

    english: str
    spanish: str
    pair_id: int
    doc_id: int
    times: int          # how often this exact rendering occurs
    english_times: int  # how often the English line occurs at all
    similarity: float
    exact: bool

    @property
    def citation(self) -> str:
        return f"pair_id {self.pair_id}, document {self.doc_id}"

    @property
    def consensus(self) -> float:
        """Share of this English line's renderings that agree with this one.

        A misaligned row is almost always a singleton among several agreeing ones, so a
        low share is the cheapest available warning that a citation may be junk.
        """
        return self.times / self.english_times if self.english_times else 0.0


def find_precedent(
    cue: str,
    *,
    limit: int = 5,
    neighbours: int = 40,
    max_length_ratio: float = 2.2,
    min_consensus: float = 0.0,
) -> list[Precedent]:
    """How have Mexican subtitlers rendered something like this English line?

    Neighbours are found by meaning, then their renderings are *grouped* rather than
    listed row by row. Grouping matters for more than tidiness: about one row in five of
    the underlying corpus is misaligned, and a misaligned row is nearly always a lone
    reading of an English line that several other rows agree on. Ranking by agreement
    pushes that noise down without needing to detect it.

    Ordering is by similarity rounded to two decimals, then by agreement: neighbours that
    are equally close should be separated by how many translators chose the reading, not
    by the fourth decimal of a cosine.

    `min_consensus` drops readings that too few translators agree with. It defaults to
    **off**, because no threshold here has been tuned against a labelled set — the lever
    is exposed so a caller can use it deliberately, not presented as a solved filter.
    A length filter drops the crudest misalignments up front; it does not catch them all
    (see CORPUS.md), so a citation still has to be verified before a person sees it.
    """
    ch = client()
    vector = embed_one(cue)

    rows = ch.query(
        """
        SELECT c.en, c.es, min(c.pair_id) AS pair_id, any(c.doc_id) AS doc_id,
               count() AS times, n.d AS distance
        FROM (
            SELECT text, cosineDistance(embedding, {vec:Array(Float32)}) AS d
            FROM mx_embeddings
            ORDER BY d ASC
            LIMIT {n:UInt32}
        ) AS n
        INNER JOIN mx_corpus AS c ON c.en = n.text
        WHERE length(c.es) <= length(c.en) * {ratio:Float64}
          AND length(c.en) <= length(c.es) * {ratio:Float64}
        GROUP BY c.en, c.es, n.d
        ORDER BY distance ASC, times DESC
        """,
        parameters={"vec": vector, "n": neighbours, "ratio": max_length_ratio},
    ).result_rows

    totals: dict[str, int] = {}
    for english, _spanish, _pid, _did, times, _d in rows:
        totals[english] = totals.get(english, 0) + times

    ranked = sorted(
        rows,
        key=lambda r: (-round(1.0 - float(r[5]), 2),
                       -(r[4] / totals[r[0]] if totals[r[0]] else 0),
                       -r[4]),
    )

    seen: set[str] = set()
    out: list[Precedent] = []
    for english, spanish, pair_id, doc_id, times, distance in ranked:
        if spanish in seen:
            continue
        if totals[english] and times / totals[english] < min_consensus:
            continue
        seen.add(spanish)
        out.append(Precedent(
            english=english, spanish=spanish, pair_id=int(pair_id), doc_id=int(doc_id),
            times=int(times), english_times=totals[english],
            similarity=round(1.0 - float(distance), 3),
            exact=english.strip().lower() == cue.strip().lower(),
        ))
        if len(out) >= limit:
            break
    return out
