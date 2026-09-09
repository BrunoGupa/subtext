# Subtext — Mexican Spanish the way subtitlers really write it, with proof

*The faithful subtitle track, for Mexican Spanish. A new line written for yours, grounded in
what Mexican subtitlers wrote — 103.6 million subtitle pairs in ClickHouse — and citable by
the line it rests on.*

**Agentic Cinema: The Blockbuster Hackathon (Google Cloud) — ClickHouse track.** Apache-2.0.

**Built on Google Cloud:** Gemini 3.8 Flash through **Google ADK** · `gemini-embedding-001`
for the vector channel · **Cloud Run** for the site · **ClickHouse Cloud** on GCP for the
corpus, the SQL and the HNSW vector search, reached at runtime through **`mcp-clickhouse`**,
the official ClickHouse MCP server.

**Live on Cloud Run:** https://subtext-475154074268.us-east1.run.app · **Video:** _YouTube link goes here_

## Two subtitle tracks, and only one of them exists

Streaming already lets you choose almost everything about a film: which one, in what audio
language, dubbed or not, with subtitles in any of a dozen languages. One editorial decision
is still made for you and offered in exactly one version — **how much of the original line
survives into the subtitle.**

That decision is not a rumour. It is measured, independently, on different films, different
target languages and different decades:

| study | film | target | what it measured |
|---|---|---|---|
| Ávila-Cabrera (2015) | *Pulp Fiction* | European Spanish | **41.8%** of the offensive/taboo load never reaches the subtitle — 27.7% omitted outright, 14.1% neutralised. Omission is the single most used strategy at **27.2%** |
| Rohmawati (2021) | *Deadpool 1 & 2* | Indonesian | deletion is the **most frequent** strategy in *Deadpool Two*: 65 of 166 instances (**39%**), 29% in the first film |
| Hawel (2020) | *The Wolf of Wall Street* | Arabic | of 506 swearing instances, **72.9%** omitted, 25.1% softened |

We then measured the same effect on our own corpus, without looking for it: of **688,351**
English lines containing *fuck*, **39.6% lose all profanity marking in Spanish**
(CORPUS.md §5). Four measurements, four corpora, one direction.

Call that the **moderated track**, and it is a legitimate product. Broadcast standards,
younger audiences and the profession's own conventions all ask for it, and the industry is
good at making it. The problem is not that it exists. The problem is that it is the only
thing on offer: a viewer who wants to know what the character actually said has nowhere to
get it.

**This builds the other track.** A faithful subtitle: the line the film actually says,
rendered in the Spanish the viewer actually speaks — Mexican, with the localisms a Mexican
translator would use. Faithful to register *and* to variety, because both are ways a
subtitle can quietly stop being the film. The viewer picks the film; the viewer should pick
this too.

Faithful is the harder half to build, which is why it does not exist yet. Asked for
"Mexican Spanish", a language model reaches for whatever *sounds* Mexican — `güey`, `órale`
— with nothing behind the choice, and intuition is a poor guide to what Mexican subtitlers
actually write. This corpus says `chamarra` **76** times against `chaqueta` **16**, so that
one is real; it also says `coche` **243** times, a word an intuition-driven filter would
have struck out as Spain's. Nor can the corpus simply be sorted by frequency to settle it,
because the corpus is itself mostly the moderated track. Two consequences follow, and the
whole system is built on them:

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

Two things this is not. It is not a filter that makes subtitles coarser: where the film is
mild the faithful rendering is mild, and fidelity to a line like `Get in the car.` is
entirely a question of which Spanish, not how strong. And it is not a claim that the
moderated track is wrong. It is a claim that one of the two should not be missing.

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

The web UI takes one English line and shows every reading with its evidence, and how long
each half took: ClickHouse Cloud for the precedents and the agreement, Gemini for the
readings. **The example lines on the page are not in the corpus** — checked one by one. An
exact match would simply hand back a subtitle somebody already wrote, so the demo lines are
ones the system has to assemble from precedent: phrases it has seen, in lines it has not.
Under every answer the page also prints two **controls**, uncited and run through the same
register lexicon: the same Gemini model asked for "Mexican Spanish" with no corpus behind
it, and Google Translate, which has no Mexican Spanish to ask for — its target is `es`, one
Spanish, and on the evaluation lines it answers in peninsular (`coche`, `chaqueta`,
`conduce`, `coño`). On mild lines Gemini alone often agrees with the corpus; it earns its
keep where register is loaded (`Get out of my car.` → `Sal de mi auto.`, cited, against the
model's `Bájate de mi carro.` and Google's `Sal de mi coche.`, where subtitlers write `auto`
4:1). The front page carries a showcase of
twenty evaluation lines, selected by us to show the range, with all four answers side by
side: Subtext with its citations, the official subtitle that shipped, Gemini alone, Google
Translate. Two evaluation pages, `/examples/paper` and `/examples/film`, show the **full**
runs, failures included, over the 40 scholar-chosen lines and the 102 film-sampled lines. A line in the URL, `/?q=Shut+up+and+drive.`, is
localised on arrival.

The question-answering commands — `subtext search`, `aggregate`, `schema` and `ask` — are
still in the CLI and still work, but the corpus their examples were written against
(*The Lighthouse Contract*, with the `Vale` character and its season structure) was
withdrawn on 2026-09-07, so those examples no longer have data behind them. Point them at a
corpus you load yourself with `subtext load`.

## How a phrase gets its Spanish, when nothing is word-aligned

The corpus is a bitext: one English line beside one Spanish line, 335,800 of them. Nowhere
in it does anyone record that `up his ass` is `por el culo`. There is no word alignment, so
asked how the corpus renders a phrase, the database can only hand back **whole lines that
contain it, with their whole Spanish** — and presenting that as the phrase's translation is
a lie in proportion to how little of the line the phrase occupies. Answer `his ass` with
`I could kick his ass.` → `Podría patearle su trasero.` and the model is handed `trasero`
as attested fact, when the corpus attested a sentence about kicking.

The bound that prevents this is coverage: quote a line only when the phrase accounts for
**60%** of it, so the line's Spanish really is the phrase's Spanish. It is correct, and on
its own it is also silent exactly where it matters. An idiom is short and lives inside long
sentences: `up his ass` is ten characters and the six corpus lines carrying it run 42 to 124,
so all six fail coverage — while six of them say `culo`. A rule that reads character lengths
and never reads the Spanish will discard unanimous evidence without seeing it.

So a second channel asks a different question. Not *"how was this line translated"* but
**"what do all the lines carrying this phrase agree on?"** Here are all six, in full, as the
database returns them:

| `pair_id` | Spanish |
|---|---|
| 53540211 | Mira, ese hijo de la chingada de nadie tiene un cohete **en el culo**… |
| 66251550 | Dile que agarre un paraguas, se lo meta **por el culo** y luego lo abra. |
| 82364436 | Dile a Tavares que se meta su respeto **por el culo**. |
| 88126807 | En su bolsa, su portafolio o **en su culo**. |
| 88128402 | ¿Y si le chingamos la otra pata a tu nana y se la metemos **por el culo**? |
| 97191204 | Sabes, Kestin tiene un palo **en el culo**. |

Six translators, six films, one rocket, one umbrella, one man's respect, one briefcase, one
severed foot and one stick. **The only thing these sentences share is the English phrase —
and `culo`.** Nobody agreed on it and no rule proposed it; the agreement is a property of the
corpus, and reading it out is a matter of counting: cut each Spanish side into its 1–4 word
runs, count how many of the six carry each run, and compare that against what the run does
across all 335,800 lines.

Which is where frequency fails a second time, for the same reason it fails at ranking
renderings. Sorted by how many lines contain them:

| Spanish run | in the phrase's lines | corpus-wide | verdict |
|---|---|---|---|
| `el` | 5 of 6 | roughly a third of everything | **4×** — noise |
| `por el culo` | 3 of 6 | 26 lines in 335,800 | **5,790×** — the rendering |

`el` wins on count and means nothing. Ranking by enrichment instead — the same choice the
register lexicon makes, for the same reason — puts `por el culo` first by three orders of
magnitude. Two SQL queries, no aligner, no model.

**It abstains, which is the part that makes it usable.** The shorter phrase `his ass` spans
48 lines that mostly mean *kick his ass*; the readings scatter, the best candidate is `su`
at 7×, the bound is 50×, and the channel says nothing at all. An agreement asserted where
there is none is worse evidence than none, because it reads as citable.

Two things it deliberately does not claim:

- **It is about wording, never grammar.** `shut up` agrees on `cállate`, and that is a fact
  about the verb, not about who is being addressed — `¡Cállate!`, `¡Cállese!` and
  `¡Cállense!` all rest on it. Form of address comes from the other channel, and an agreed
  rendering never introduces a subject pronoun.
- **It ranks below a precedent for the line itself.** `what the fuck` agrees on
  `qué chingados` at 301×, and the corpus also renders that exact line as `¿Qué pedo?`
  eighteen times. The line wins; the agreement becomes a check on it.

Everything above is retrieved, and the distinction that matters is between a rule and an
answer. The standing instruction does carry Spanish — it forbids `vosotros`, it names the
peninsular words Mexican subtitlers do not write, and it says *not* to avoid `coche` (373
times in this corpus) or `vale` (924), and *not* to reach for `güey` where no evidence shows
it. Those are grammar and negative guidance, derived from the corpus and stated once.

What never appears there is the **answer to a cue the system is measured on**. During
development the rule about coarse wording was written with its example inline — `up his ass`
agrees on `por el culo` — which meant the model was handed the answer to the exact case the
channel was being judged on, and the retrieval could have been broken without the output
changing. The example was removed and the rule left in the abstract; a test fails if any of
those words returns to a static prompt
([`tests/test_variants.py`](tests/test_variants.py)). The check is a guard for the phrases
under measurement, not a proof about the whole lexicon.

What *is* chosen by hand is four numbers: the 60% coverage bound, the 50× enrichment bound,
a minimum of 5 attesting lines, and a 400-line sample per phrase. The 50× was swept over the
207-phrase evaluation set — hit rate against the human translations is flat from 10× to 75×
and falls above it, so the bound sits where coverage is highest without paying for it.

Every line that survives all of this is marked in the output: a reading backed by an agreed
rendering says so, and a reading whose nearest precedent is too far away is flagged `WEAK`
rather than shipped with a `pair_id` that does not support it.

## How it works

```
English cue
   │
   ├─ address tagger ......... which readings the corpus attests for this line:
   │                           tú / usted / ustedes, or one unmarked reading. Cached
   │                           in ClickHouse — the label belongs to the line, not
   │                           to the query, so it is paid for once
   │
   ├─ TWO retrieval channels, always both, they fail in opposite directions
   │    ├─ phrase index ...... exact 1–4 word n-grams (171,888 of them), plus the
   │    │                      AGREED RENDERING measured across every line carrying
   │    │                      the phrase. Silent on wording the corpus has not seen
   │    └─ vector index ...... HNSW over 269,869 embeddings of distinct English lines.
   │                           Always answers something, which is exactly why its
   │                           answer is labelled weaker in the prompt
   │
   ├─ translator ............. ONE Gemini call per reading. It receives the evidence
   │                           and the grammar of the reading it is writing, and may
   │                           refuse a reading the line rules out (usted to a child)
   │
   └─ gates, in Python ....... register against the corpus lexicons · form of address
                               re-tagged and compared · well-formedness. One retry,
                               naming what was wrong and forbidding anything else
                               from changing
```

**The loop is closed by Python, never by Gemini.** Nothing asks the model whether it is
happy with its own output, and nothing lets it decide whether to retrieve. That follows
from the project's own rule: a step that makes no creative decision should not be an
`LlmAgent`. Retrieval looks things up and the gates count words against the same lexicons
that built the corpus — only the translation is judgement.

The vector half is one statement. Neighbours are found by meaning, then their renderings
are **grouped rather than listed**, because roughly one row in five of the underlying
corpus is misaligned and a misaligned row is nearly always a lone reading of a line several
other rows agree on. Ranking by agreement pushes that noise down without having to detect
it ([`retrieval.py`](src/subtext/retrieval.py)):

```sql
SELECT c.en, c.es, min(c.pair_id) AS pair_id, count() AS times, n.d AS distance
FROM (
    SELECT text, cosineDistance(embedding, {vec:Array(Float32)}) AS d
    FROM mx_embeddings ORDER BY d ASC LIMIT {n:UInt32}        -- HNSW: 6 ms, not 41
) AS n
INNER JOIN mx_corpus AS c ON c.en = n.text
WHERE length(c.es) <= length(c.en) * {ratio:Float64}          -- drop crude misalignments
  AND length(c.en) <= length(c.es) * {ratio:Float64}
GROUP BY c.en, c.es, n.d
ORDER BY distance ASC, times DESC
```

Every reading that comes out carries what backed it: the `pair_id`s it can be checked
against, whether an agreed rendering stood behind it, and a `WEAK` mark when the nearest
precedent was too far away to support the citation printed beside it. A guess and a
citation are never rendered the same way.

### The two ADK agents

`agent/localiser.py` is the localisation pipeline above as an ADK `LoopAgent` —
`EvidenceAgent` (`BaseAgent`, no tokens) → translator (`LlmAgent`, one call) →
`RegisterGate` (`BaseAgent`, no tokens), `max_iterations=2`. `agent/agent.py` is the
question-answering agent, an `LlmAgent` that loops over tools reached through the official
`mcp-clickhouse` MCP server — correctly a tool loop, because there the choice of tool is
the work.

**Honest note:** `subtext localise` and the web UI currently call the equivalent pipeline
in [`variants.py`](src/subtext/variants.py) directly rather than
`localiser.localise_async`, which is reachable only through its own `__main__`. The two
implement the same three steps and the same Python-closed loop; consolidating them onto
the ADK path is open work.

## Measured, not asserted

**The retrieval evaluation has been withdrawn (2026-09-07).** It ran against a synthetic
sample corpus generated for this repository, and its "hand-labelled" golden set was labelled
by the same process that wrote the dialogue. Numbers produced that way measure self-consistency,
not retrieval quality, so they have been removed rather than restated with a caveat.

What replaces it is measured against text this project did not write. The full pipeline was
run over the **paper** group on 2026-09-09 — 40 lines chosen by published translation
scholars as cases of difficulty, 37 of them absent from the corpus:

| | |
|---|---|
| lines translated | **40 of 40**, none refused for content |
| readings written (tú / usted / ustedes / unmarked) | 57 |
| readings grounded in a cited `pair_id` | 56 |
| readings flagged `WEAK` (precedent too far to support the citation) | 7 |
| readings failing the register gate (peninsular Spanish) | **0** |
| lines with no surviving reading | 1 (*And I will strike down upon thee* — the model declined every form of address) |
| median wall time per line | 13.6 s, of which ~4 s is ClickHouse Cloud |

The 102 **film-sampled** lines (none in the corpus) ran the same day: 145 readings, 16
weak, 0 peninsular leaks, 3 lines with no surviving reading, 0 errors.

`Shut the fuck up` → `Cierra la boca.` where the official Spanish subtitle has `cierra el
pico`; `What the fuck do you think I'm doing?` → `¿Qué chingados crees que estoy haciendo?`
where the official subtitle drops the profanity entirely (`¿Qué crees que hago?`). The 102
film-sampled lines run the same way; both sets are rendered line by line, with every
citation, on the site's evaluation pages. The official subtitles in the paper group are
European Spanish, printed for comparison, not as the right answer.

## Stack
- **Gemini + Google ADK** — planning, tool selection, self-validation (hackathon requirement #1)
- **ClickHouse**, reached two ways — the corpus, the aggregation *and* the vector search
  (partner product, requirement #2):
  - **`mcp-clickhouse`**, the official MCP server, wired in as an ADK `McpToolset`. The agent's
    SQL goes through it, which is what the ClickHouse track requires.
  - **`clickhouse-connect`** for the vector path, which has no choice: the question must be
    embedded in Python before there is a query to send.
- **`gemini-embedding-001`** — 768-dimension vectors for the semantic channel, one per
  distinct English line, normalised in code as Google requires below 3072 dimensions
- **Python 3.12 + uv**

## The corpus

**`data/sample/` was removed on 2026-09-07.** It shipped *The Lighthouse Contract*, ~200
invented lines written for this repo, and it was withdrawn along with the evaluation that
used it — the dialogue and its "hand-labelled" golden set came from the same process, so any
score over it measured self-consistency. What ships instead is
[`data/mx_docs.tsv`](data/mx_docs.tsv): 241 document boundaries as `pair_id` ranges, integers
only, no text, from which `subtext build-mx` reconstructs the Mexican corpus in about 40
seconds against the corpus you fetched yourself.

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

## Deploying the site

The web UI is one container ([`Dockerfile`](Dockerfile)) with no state of its own: the corpus
and the embeddings live in ClickHouse Cloud, the translations come from Gemini, and both are
reached with environment variables — the same names as `.env.example`, set on the service
rather than in a file. On Cloud Run:

```bash
gcloud run deploy subtext --source . --region us-east1 --allow-unauthenticated \
  --set-env-vars CLICKHOUSE_HOST=...,CLICKHOUSE_HTTP_PORT=8443,CLICKHOUSE_SECURE=true,\
CLICKHOUSE_USER=...,CLICKHOUSE_DATABASE=subtext,SUBTEXT_MODEL=gemini-3.8-flash \
  --set-secrets CLICKHOUSE_PASSWORD=clickhouse-password:latest,GOOGLE_API_KEY=gemini-key:latest \
  --max-instances 1 --memory 1Gi
```

`make deploy GCP_PROJECT=...` runs the same command with the `.env` values. `--max-instances
1` is deliberate: the per-address rate limit and the daily Gemini budget in
[`app.py`](src/subtext/web/app.py) are held in memory, so one instance is what makes them
real. The site is public, and every request is paid Gemini tokens. Answers are cached in a
`web_cache` table in ClickHouse as well as in memory, so a redeploy or a scale-to-zero does
not pay for the same line twice, and the page says which cache it came from. The Google
Translate control needs the Cloud Translation API enabled and `roles/cloudtranslate.user`
on the service account; without them it is simply not shown.

## Status
The corpus, both retrieval channels, the localisation pipeline, the web UI and the
evaluation over the scholar-chosen lines all run end to end. Open work, in order: consolidate
the CLI and web onto the ADK `LoopAgent` path; run the remaining evaluation groups; a shared
store for the rate limit if the site ever runs on more than one instance.

## Docs
| File | What |
|---|---|
| [`evals/test-set-sources.md`](evals/test-set-sources.md) | ⭐ Where every evaluation phrase comes from, with the citation |
| [`evals/README.md`](evals/README.md) | What is measured, and what was withdrawn on 2026-09-07 |
| [`demo/README.md`](demo/README.md) | Cited renderings for a reader who will run nothing |
| [`CORPUS.md`](CORPUS.md) | How the corpus is fetched and built, and §6: what the pipeline cannot do |
