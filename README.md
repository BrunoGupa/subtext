# Reel Query — a text-to-SQL + hybrid-retrieval agent over a film corpus

**Agentic Cinema: The Blockbuster Hackathon (Google Cloud) — ClickHouse track.**
Deadline **Sept 9, 2026, 2:00 PM PDT**.

An agent that answers questions about a film/TV corpus that could not be answered any other
way — by planning a query, running it against ClickHouse, and validating its own answer
against the data before it speaks.

The demo question that defines the project:

> *"How many times across six seasons does this character break a promise?"*

There is no `broke_a_promise` column. SQL alone cannot answer it, and neither can semantic
search alone — you need **both**: vector similarity finds candidate lines, SQL aggregates and
filters them by season, character and timecode.

It is also, measurably, the hardest question in the golden set — `line`-level chunking misses
it entirely at k=20, and [`evals/results.md`](evals/results.md) §4 explains exactly why and
what would fix it. That writeup is the part of this repo worth reading first.

## Quick start

Needs Docker and [uv](https://docs.astral.sh/uv/). No accounts, no downloads, no API key for
everything except the agent itself.

```bash
make demo      # .env + ClickHouse + deps + schema + corpus + embeddings
make eval      # recall@k and the abstention curve over the golden set
make sweep     # 3 chunk strategies x 5 values of k
```

Then, without any API key:

```bash
# vector search over dialogue
uv run reel-query search "someone refuses to lie for him" -k 5

# the hybrid path: vector search feeding a SQL GROUP BY, in one query
uv run reel-query aggregate "You gave me your word and then you did not do it" \
    --involving Vale --group-by season --strategy window --window-size 3 \
    --max-distance 0.75 -k 60 --show-sql

# the schema slice a question would put in front of the SQL generator
uv run reel-query schema "which season has the most broken promises?"
```

And with a [free Gemini key](https://aistudio.google.com/apikey) in `.env`:

```bash
uv run reel-query ask "How many times does Vale break a promise?" --trace
```

## How it works

```
question
   │
   ├─ inspect_schema ............ embed the question, retrieve only the relevant
   │                              columns (12 of 33) into the SQL-generation prompt
   ├─ search_dialogue ........... cosineDistance over Array(Float32) in ClickHouse
   ├─ aggregate_semantic_matches  ⭐ ONE query: a vector-search CTE feeding a GROUP BY
   ├─ run_sql .................... model-written SELECT, read-only guarded
   └─ verify_answer ............. every cited line checked against what was actually
                                  retrieved, before the answer is spoken
```

The hybrid query is one statement, not two features glued together
([`retrieval.py`](src/reel_query/retrieval.py)):

```sql
WITH candidates AS (
    SELECT chunk_id, line_ids, cosineDistance(embedding, {q:Array(Float32)}) AS distance
    FROM line_chunks WHERE strategy = {strategy:String} ORDER BY distance ASC LIMIT {pool:UInt32}
)
SELECT l.season, count(DISTINCT l.line_id) AS matches, groupArray(4)(l.text) AS examples
FROM candidates AS c
ARRAY JOIN c.line_ids AS lid
INNER JOIN lines AS l ON l.line_id = lid
WHERE (lower(l.character) = lower({f_inv:String})      -- the speaker, or
   OR  lower(l.speaks_to) = lower({f_inv:String}))     -- the person spoken to
  AND distance <= {max_distance:Float32}
GROUP BY l.season ORDER BY matches DESC
```

Vector search finds; SQL counts. Neither half answers the question alone.

## Measured, not asserted

34 hand-labelled questions, 179 labelled line ids. Full tables and the failure analysis in
[`evals/results.md`](evals/results.md).

| strategy | recall@10 | recall@20 | hit@10 | hit@20 |
|---|---|---|---|---|
| `line` | **0.510** | **0.650** | 0.800 | 0.833 |
| `window3` | 0.517 | 0.607 | **0.900** | **0.933** |
| `window5` | 0.479 | 0.544 | 0.867 | 0.933 |

The two metrics disagree, and the disagreement is the point: windows reliably find the right
*scene* and then spend the line budget on its neighbours. Which strategy is better depends on
whether a model reads the context or a `GROUP BY` counts it.

The distinction the eval is built around is **retrieval failure** (the answer never came back
— recall@k) versus **generation failure** (it came back and the answer ignored it —
faithfulness). Different bug, different fix, and one blended accuracy number would hide which
one you have.

## Stack
- **Gemini + Google ADK** — planning, tool selection, self-validation (hackathon requirement #1)
- **ClickHouse** — the corpus, the aggregation, *and* the vector search (partner product, requirement #2)
- **sentence-transformers** (MiniLM, local) — embeddings, $0
- **Python 3.12 + uv**

## The corpus

`data/sample/` ships *The Lighthouse Contract* — six invented seasons of a salvage-crew
series, ~200 lines, written for this repo and released under its Apache-2.0 licence. It exists
so the demo runs on a clean clone with no downloads and no licensing questions, and because
the golden set needs promises that are genuinely broken later on.

For real scale, point the loader at subtitles you obtained yourself:

```bash
uv run reel-query load --source srt --path ./data/raw/show --title "Some Show" --title-id tt1234567
uv run reel-query embed --strategy window --window-size 3
```

## Safety

The agent writes SQL and then runs it, which is the point of the project and also the obvious
way to get hurt once it is publicly reachable. Generated statements go through
[`sql_guard.py`](src/reel_query/sql_guard.py) — single statement, `SELECT`/`WITH` only,
comments and string literals stripped before keyword checks — *and* ClickHouse runs them with
`readonly=1`, a time limit and a row cap. Config is entirely environment-driven; nothing in
this repo carries a credential.

## Status
🔒 **PRIVATE while under construction.** Goes public before submission — the hackathon requires
a public repo under an OSI licence (Apache-2.0, already in `LICENSE`) and a publicly hosted URL.

## Docs
| File | What |
|---|---|
| [`evals/results.md`](evals/results.md) | ⭐ The numbers and the failure analysis |
