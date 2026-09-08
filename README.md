# Subtext — the Spanish a Mexican subtitler actually wrote

*Retrieval-grounded localisation of film dialogue into Mexican Spanish, over 103.6 million
subtitle pairs in ClickHouse.*

**Agentic Cinema: The Blockbuster Hackathon (Google Cloud) — ClickHouse track.**
Deadline **Sept 9, 2026, 2:00 PM PDT**. Apache-2.0.

## The problem, and it is documented

Ask any language model to translate a line into "Mexican Spanish" and it will hand you a
stereotype: `carro`, `chamarra`, `güey`, whether or not a Mexican translator would have
written them. Ask a subtitle corpus and you get something different — but not what you
might expect, because **subtitling deletes swearing as a matter of professional routine.**

That is not our claim. It is the finding of the audiovisual translation literature, measured
independently on different films, different target languages and different decades:

| study | film | target | what it measured |
|---|---|---|---|
| Ávila-Cabrera (2015) | *Pulp Fiction* | European Spanish | **41.8%** of the offensive/taboo load never reaches the subtitle — 27.7% omitted outright, 14.1% neutralised. Omission is the single most used strategy at **27.2%** |
| Rohmawati (2021) | *Deadpool 1 & 2* | Indonesian | deletion is the **most frequent** strategy in *Deadpool Two*: 65 of 166 instances (**39%**), 29% in the first film |
| Hawel (2020) | *The Wolf of Wall Street* | Arabic | of 506 swearing instances, **72.9%** omitted, 25.1% softened |

We then measured the same effect on our own corpus, without looking for it: of **688,351**
English lines containing *fuck*, **39.6% lose all profanity marking in Spanish** (CORPUS.md
§5). Four measurements, four corpora, one direction.

So a subtitle corpus tells you what the profession *permits*, not what the language *has*.
Two consequences follow, and the whole system is built on them:

**1. Frequency is the wrong ranking.** `¡Jódete!` is the most frequent rendering of
`Fuck you!` inside our Mexican corpus, and it is **0.9× enriched** — it appears *less* often
in Mexican productions than in Spanish subtitles at large. `¡Chinga tu madre!` is rarer there
and **188×** enriched. Ranking by frequency reproduces the convention; ranking by enrichment
against the full 103.6M lines recovers what is Mexican.

**2. There is no single right answer.** English `you` is tú, usted and ustedes at once, and a
subtitle line arrives with no scene attached. `Get in the car.` appears in this corpus as
`Súbete al coche.`, `Súbase.` and `- Entren al carro.`, in three different films. Returning
one line means choosing silently, by accident. So this returns **every reading the corpus
attests**, each grounded in its own evidence and each citable by `pair_id`:

```
$ uv run subtext localise "Shut up!"
EN  Shut up!

ES  ¡Cállate!    [tú]        cited: pair_id 13142143, 83689621, 83067397
ES  ¡Cállese!    [usted]     cited: pair_id 84948787, 81922469, 2911163
ES  ¡Cállense!   [ustedes]   cited: pair_id 12557894, 63519068, 88388771
```

Nothing there was written by a model. Each line was written by a Mexican subtitler, and the
`pair_id` says which one.

## How it is evaluated, and who chose the phrases

A test set chosen by the people who built the system proves nothing, so the origin of every
phrase travels with its row — see [`evals/test-set-sources.md`](evals/test-set-sources.md).
The strongest group comes first: **lines selected by published translation scholars as cases
of difficulty**, with the official Spanish subtitle the paper prints, and the citation
attached. Below them, phrases chosen by the American Film Institute; then lines selected by
rendering entropy over the **full** corpus rather than over the Mexican slice, which would
have been circular; every one verified absent from the corpus this system retrieves from.

Lines the system cannot answer stay in the set on purpose. Coverage only means something if
the failures are counted with it.

## Quick start

Needs Docker and [uv](https://docs.astral.sh/uv/). The corpus is fetched from its original
host at run time and never redistributed here — see CORPUS.md §1.

```bash
make demo                              # .env + ClickHouse + deps + schema
make fetch                             # the OPUS + IMDb corpora (~4.2 GB, one time)
make corpus                            # build the Mexican corpus + embeddings (~1.5 min)
make localise L="What the fuck?"       # one English line, every Mexican reading
make serve                             # the web UI on http://127.0.0.1:8000
```

`subtext localise --evidence-only` shows what was retrieved without writing a translation,
so the retrieval can be inspected on its own.

The UI has three tabs. **Agent** needs a Gemini key; **Hybrid count** and **Semantic search**
do not, and the app opens on the hybrid tab when no key is set. Every tab shows its working —
the SQL that ran, the lines that came back with their timecodes, and, for the agent, the tools
it chose and whether its own self-check passed. That trace is the demo; the answer alone is
the least interesting part.

Then, without any API key:

```bash
# vector search over dialogue
uv run subtext search "someone refuses to lie for him" -k 5

# the hybrid path: vector search feeding a SQL GROUP BY, in one query
uv run subtext aggregate "You gave me your word and then you did not do it" \
    --involving Vale --group-by season --strategy window --window-size 3 \
    --max-distance 0.75 -k 60 --show-sql

# the schema slice a question would put in front of the SQL generator
uv run subtext schema "which season has the most broken promises?"
```

And with a [free Gemini key](https://aistudio.google.com/apikey) in `.env`:

```bash
uv run subtext ask "How many times does Vale break a promise?" --trace
```

## How it works

```
question
   │
   ├─ inspect_schema ............ embed the question, retrieve only the relevant
   │                              columns (12 of 33) into the SQL-generation prompt
   ├─ search_dialogue ........... cosineDistance over Array(Float32) in ClickHouse
   ├─ aggregate_semantic_matches  ⭐ ONE query: a vector-search CTE feeding a GROUP BY
   ├─ run_query ................. model-written SELECT, executed by the official
   │                              ClickHouse MCP server (`mcp-clickhouse`), read-only
   └─ verify_answer ............. every cited line checked against what was actually
                                  retrieved, before the answer is spoken
```

The hybrid query is one statement, not two features glued together
([`retrieval.py`](src/subtext/retrieval.py)):

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

**The retrieval evaluation has been withdrawn (2026-09-07).** It ran against a synthetic
sample corpus generated for this repository, and its "hand-labelled" golden set was labelled
by the same process that wrote the dialogue. Numbers produced that way measure self-consistency,
not retrieval quality, so they have been removed rather than restated with a caveat.

What replaces it is measured against text this project did not write: the Spanish side of a
human-translated subtitle corpus, used as a reference translation. That work is in progress and
its numbers will appear here when they exist.

## Stack
- **Gemini + Google ADK** — planning, tool selection, self-validation (hackathon requirement #1)
- **ClickHouse**, reached two ways — the corpus, the aggregation *and* the vector search
  (partner product, requirement #2):
  - **`mcp-clickhouse`**, the official MCP server, wired in as an ADK `McpToolset`. The agent's
    SQL goes through it, which is what the ClickHouse track requires.
  - **`clickhouse-connect`** for the vector path, which has no choice: the question must be
    embedded in Python before there is a query to send.
- **sentence-transformers** (MiniLM, local) — embeddings, $0
- **Python 3.12 + uv**

## The corpus

`data/sample/` ships *The Lighthouse Contract* — six invented seasons of a salvage-crew
series, ~200 lines, written for this repo and released under its Apache-2.0 licence. It exists
so the demo runs on a clean clone with no downloads and no licensing questions, and because
the golden set needs promises that are genuinely broken later on.

For real scale, point the loader at subtitles you obtained yourself:

```bash
uv run subtext load --source srt --path ./data/raw/show --title "Some Show" --title-id tt1234567
uv run subtext embed --strategy window --window-size 3
```

## Safety

The agent writes SQL and then runs it, which is the point of the project and also the obvious
way to get hurt once it is publicly reachable. Generated statements go through
[`sql_guard.py`](src/subtext/sql_guard.py) — single statement, `SELECT`/`WITH` only,
comments and string literals stripped before keyword checks — *and* ClickHouse runs them with
`readonly=1`, a time limit and a row cap. The MCP server adds a third, independent layer — it
connects as a read-only user, so a `DROP` that somehow got past the guard still fails at the
server. Config is entirely environment-driven; nothing in this repo carries a credential.

## Status
🔒 **PRIVATE while under construction.** Web UI, retrieval and evaluation are done; the agent
loop is built but unrun (needs a Gemini key), and hosting is not decided. Goes public before submission — the hackathon requires
a public repo under an OSI licence (Apache-2.0, already in `LICENSE`) and a publicly hosted URL.

## Docs
| File | What |
|---|---|
| [`evals/results.md`](evals/results.md) | ⭐ The numbers, the sweep, and the failure analysis |
| [`evals/README.md`](evals/README.md) | How the golden set is labelled and what each metric measures |
| [`data/sample/README.md`](data/sample/README.md) | The bundled corpus, and how to load your own |
