# The parallel corpus

*How subtitle translation is analysed here: what the data is, how to get it, what the
tables mean, and — importantly — what this pipeline **cannot** tell you.*

The dialogue corpus in `data/sample/` is synthetic and ships with the repo. This
document is about the other corpus: **103 million real English↔Spanish subtitle pairs**,
which does *not* ship with the repo and never will.

---

## 1. Why the data is not in this repository

OpenSubtitles text is contributed by users and OPUS is explicit that it does not own
it:

> We do not own any of the text from which the data has been extracted. We only offer
> files that we believe we are free to redistribute. If any doubt occurs about the
> legality of any of our file downloads we will take them off right away.

Re-hosting 2.5 GB of that from a public portfolio repository would be careless. So the
**loader** is the deliverable and the data is fetched from its original host at run
time — the arrangement `torchvision.datasets(download=True)`, `nltk.download()` and the
HuggingFace dataset scripts all use.

Everything downloaded lands in `data/raw/`, which is gitignored, along with `*.srt`,
`*.ass`, `*.vtt` and friends so no third-party subtitle can reach the repo by accident.

### Licence obligations you inherit

**OpenSubtitles / OPUS.** The README inside the archive asks, in these words:

> Please, add a link to http://www.opensubtitles.org/ to your website and to your
> reports and publications produced with the data!

and to cite:

> P. Lison and J. Tiedemann, 2016. *OpenSubtitles2016: Extracting Large Parallel
> Corpora from Movie and TV Subtitles.* LREC 2016.

**IMDb.** The [bulk datasets](https://developer.imdb.com/non-commercial-datasets/) are
free for **personal and non-commercial use** with attribution to IMDb. If this project
ever becomes commercial, that dependency has to be replaced.

These are obligations, not courtesies. Honour them in anything you publish.

---

## 2. Getting the data

```bash
subtext fetch              # everything, ~4.2 GB, prompts first
subtext fetch opus         # just the sentence pairs + alignment
subtext fetch imdb         # just the title metadata
subtext fetch --lang-pair en-fr opus     # a different language pair
```

It prints every file with its size and licence terms and waits for confirmation.
Downloads go to a `.part` file and are renamed only on success, so an interrupted run
never leaves a truncated archive looking complete.

| Dataset | Size | Contents |
|---|---|---|
| `opus-en-es-moses` | 2.5 GB | 105,482,431 aligned sentence pairs |
| `opus-en-es-align` | 1.0 GB | Year + IMDb id for all 135,470 documents |
| `imdb-basics` | 216 MB | 12.7M titles: name, type, year, genres |
| `imdb-akas` | 489 MB | 59.2M regional titles with `region` and `language` |

The two OPUS files are **checksum-pinned** (v2024 is a frozen release). The IMDb files
are **deliberately not** — IMDb regenerates them daily, so a pinned hash would fail for
everyone tomorrow. Verify those by row count instead.

---

## 3. Loading

```bash
subtext init-db                     # creates aligned_lines among the other tables
python -c "from pathlib import Path; from subtext.ingest.parallel import load_parallel; \
           load_parallel(Path('data/raw/opus/en-es.txt.zip'))"
```

`load_parallel` streams both sides straight out of the zip, so the 2.5 GB archive never
becomes 7.2 GB of text on disk. Full load: **3 min 31 s** for 103.6M rows.

Three filters drop 1.75% of rows as junk: either side empty or under 2 characters,
either side over 400 characters (subtitle cues are short — longer means an alignment
artefact), or both sides byte-identical (untranslated boilerplate: song titles, names,
numerals).

### Sampling

`stride=N` keeps every Nth pair. **Use it rather than `limit` alone.** The corpus is
ordered by year, so a contiguous head-of-file slice is a sample of *silent and
pre-1950 cinema*. Taking the first 2M pairs finds 35 lines containing "fuck"; striding
across the whole corpus finds them at 325× that rate. A strided 4M sample reproduced
the full corpus's headline ratio to within 2%.

---

## 4. The tables

### `aligned_lines` — 103,637,005 rows

| Column | Notes |
|---|---|
| `pair_id` | Position in the bitext |
| `corpus`, `lang_pair`, `source_lang`, `target_lang` | Constants stamped at load |
| `source_text`, `target_text` | The aligned sentences |

Both text columns carry a `tokenbf_v1` bloom-filter index, which is what keeps word
lookups fast across 103M rows.

### `corpus_films` — 135,470 rows

Built by `subtext.ingest.opus_align` from the alignment file.

| Column | Notes |
|---|---|
| `imdb_id` | The film, or for TV the **episode** |
| `series_id` | The parent series; empty for films |
| `season`, `episode` | TV only |
| `year` | 0 when the path carries a nonsense value |
| `start_line`, `n_lines` | **See the caveat in §6** |

The OPUS folder name is one of two shapes, and both decode:

```
0047478                 -> a film,  tt0047478
3276470_2741602_1_19    -> episode  tt3276470
                           series   tt2741602 (The Blacklist), season 1, episode 19
```

Composition: **91,762 TV episodes** across 5,045 series (54.6% of lines) and
**43,148 films** (45.1%). 543 documents — 0.4% — fail to resolve.

### `phrase_index` — 372,575 rows, 5.12 MB

Built by `sql/phrase_index.sql` from `mx_corpus` in about a second. That file is
**generated** — regenerate it with `python -m subtext.tokenizer > sql/phrase_index.sql`
after any change to the tokenizer, and rebuild. Every English line is
cut into all its runs of 1–4 words; each run is counted, and runs seen fewer than 3 times
are dropped.

| Column | Notes |
|---|---|
| `ng` | The phrase, lowercased |
| `n` | How many words it holds, 1–4 |
| `support` | How many lines contain it |
| `ex_pair_id` | One example line, to join back to `mx_corpus` |

This is the run-time lookup table: given an English cue, it finds attested Mexican
precedent without scanning a million rows per subtitle. The retrieval unit is the
**phrase, not the line** — of ~68.3M distinct English lines only 0.03% have any Mexican
rendering, so whole-line lookup returns nothing ~999 times in 1000. Phrases recur.

#### Tokenization: an apostrophe inside a word is kept

`What's up, dude?` tokenizes to `[what's, up, dude]`. The regex is
`[a-z]+(?:'[a-z]+)?` applied to the lowercased line.

**Do not "simplify" this by deleting the apostrophe.** An earlier build did
(`replaceAll(en, '\'', '')`), which merges contractions into unrelated English words:

| Merged key | Is really |
|---|---|
| `were` 22,951 | `we're` 10,499 **+** `were` 12,454 |
| `its` 41,163 | `it's` 40,006 + `its` 1,138 |
| `ill` 14,845 | `i'll` 14,664 + `ill` 189 |
| `hell` 4,376 | `he'll` 1,321 + `hell` 3,056 |

`we're` is the worst: a near 50/50 blend of two meanings that take different Spanish
(*somos/estamos* vs *eran/estaban*). Precedent retrieved under that key is evidence for
nothing.

**Why not the Penn Treebank convention?** PTB (Marcus et al. 1993) — what NLTK, CoreNLP,
spaCy and the Moses tokenizer implement — splits the clitic but keeps its apostrophe:
`what's` -> `what` + `'s`. That also avoids the collision, and it is the more standard
choice. We keep the token whole instead because:

1. The retrieval unit is a surface phrase capped at n=4. Splitting spends that budget on
   grammar: `what's up dude` is 3 words but 4 PTB tokens, so a 4-gram covers less text.
2. A bare `'s` is near the top of the English frequency list and carries almost no
   retrieval signal. It would dominate the `n=1` table.
3. We match phrasing precedent, not syntax. Nothing downstream parses.

**The rule that matters more than the choice:** the query path must tokenize *identically*
to the build (Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, §2.2).
If lookup re-implements this even slightly differently, every contraction silently misses —
no error, no empty result, just quietly worse translations.

So the rule lives in exactly one place, `src/subtext/tokenizer.py`. `tokens()` is the Python
side; `clickhouse_tokens_sql()` derives the SQL expression from the same pattern literal, and
`phrase_index_sql()` generates the whole build file. The two sides are checked against each
other on 3,000 random corpus rows: 0 mismatches. `tests/test_tokenizer.py` pins the behaviour,
including one case per contraction collision listed above.

**Scope:** the regex matches the ASCII apostrophe only. That is safe for this corpus —
`mx_corpus` holds 239,944 lines with an ASCII apostrophe and **zero** typographic ones (U+2019), because
OPUS normalized them. Incoming `.srt` files are not normalized, so `normalize()` folds the
five variants that turn up in subtitle files (`’ ʼ ′ \` ´`) onto `'` before matching.

### `mx_corpus` — 335,800 lines, and `mx_docs` — 241 documents

The Mexican-Spanish subset, found by scoring the corpus for register markers and keeping
the dense runs. Built by `sql/` + `src/subtext/boundaries.py`; `mx_docs` gives each
document's `pair_id` range, `mx_corpus` every line inside those ranges.

#### Boundaries are refined, not grid-aligned

The detector scores fixed 1000-line blocks, so a document's edge was only known to
±1000 lines. 523 of the 697 documents were a *single* block — both edges unknown inside
the same 1000 lines. **84.7% of the old corpus sat in such an edge zone**, so this was a
correction, not a polish.

Each edge is now re-tested at shrinking windows with falling thresholds — 10 markers per
1000 lines, then 7 per 500, 3 per 250, 2 per 125, 1 per 50 — extending outward while a
neighbouring window still passes, then eroding inward while the leading one fails. The
threshold has to fall with the window because density does not survive bisection: 1000
lines carry ~45 markers, 50 lines carry ~2.

Measured against the grid build it replaces:

| | grid | refined |
|---|---|---|
| lines | 1,028,000 | **718,925** (−30%) |
| Mexican markers | 30,740 | **32,074** (+4.3%) |
| marker density | 29.9 / 1k | **44.6 / 1k** (+49%) |
| peninsular contamination | 0.485% | **0.459%** |
| gold marker retention | 73.0% | **76.0%** |
| median document | 1,000 (grid artefact) | **700** |

⚠️ **"Gold recall" was the wrong name for that row, before any of the rest went wrong.**
Every word in `GOLD` is also in `MEXICAN`, so the number measures how much marked content
the document threshold *keeps* — it says nothing about whether the detector generalises to
markers it has never seen. It is retention, not recall, and it was never a held-out set.
The code has said so in a comment since the set was written; the README and the session
notes did not.

⚠️ **Those densities are inflated and the table is kept only to show the refinement's
effect.** Both columns were measured with a lexicon that contained five strings which were
not markers at all — see §6.4. Re-measured with the corrected lexicon, the refined corpus
scored **21.9 markers / 1k**, not 44.6. The current corpus is in the next table.

Fewer lines but *more* markers, because refinement extends as well as trims — it recovers
Mexican content the 1000-line grid cut off. Real subtitle documents run p25 500 / median
703 / p75 942 lines (`corpus_films`), which the refined distribution now matches; the grid
put 523 documents at exactly 1,000.

**Why this chain.** 156 monotone chains were swept. The ten best produce **identical
boundaries for the median document** — the edges are a property of the corpus, not of the
thresholds. That agreement is the evidence the method works; the specific numbers are not
load-bearing. Chains that trim harder (7/4/3/2 at the same windows) over-erode: median
575 lines, well under any real film.

#### Building it, and the shipped index

One command rebuilds every table above from a loaded `aligned_lines`:

```bash
uv run subtext build-mx                                  # ~75 s, one full scan
uv run subtext build-mx --from-index data/mx_docs.tsv    # ~60 s, skips the detection
```

The lexicons, the threshold and the refinement chain all live in
`src/subtext/ingest/mexican.py`, so the build is reproducible from the repository alone —
it does not depend on a notebook or a shell history. Both forms produce the same tables:
241 documents, 335,800 lines, 18,218 Mexican markers, 171,888 phrases.

#### What the corpus is now, and what it was this morning

The detector was audited on 2026-09-07 by reading the documents it had selected, smallest
and largest first. It had been trusting five strings that are not markers, and the corpus
halved when they were removed:

| | before the audit | after |
|---|---|---|
| documents | 697 | **241** |
| lines | 718,925 | **335,800** |
| marker density (corrected lexicon, both sides) | 21.9 / 1k | **54.3 / 1k** |
| Mexican : peninsular markers | 4.9 : 1 | **9.5 : 1** |
| documents under 300 lines | 79 | **8** |
| smallest document | 25 lines | **100 lines** |

Half the size and two and a half times the density: what left was what the false markers
were holding up. The details are in §6.3.

**The retention number barely moved: 76.0% → 74.3%.** Slightly over half the corpus was
removed and three quarters of the marked content stayed, which is the same fact the density
reports from the other side — the half that went carried almost no real markers.

And the enrichment figures roughly doubled, because enrichment is a rate and the
denominator shrank while the numerator did not:

| phrase | before the audit | after |
|---|---|---|
| `no mames` | 131× | **277×** |
| `¿qué pedo?` | 124× | **263×** |
| `chinga tu madre` | 89× | **188×** |
| `carajo` | 2.9× | **4.5×** |
| `jódete` | 1.1× | **0.9×** |
| `maldición` | 1.2× | **0.6×** |
| `maldita sea` | 0.7× | **0.2×** |

Read the bottom half, not the top: **`jódete` and `maldita sea` are now *below* 1**, meaning
they appear *less* often in Mexican productions than in Spanish subtitles at large. Ranking
renderings by raw frequency inside the corpus puts `¡Jódete!` first for `Fuck you!`, because
subtitling neutralises profanity by convention — 39.6% of English `fuck` lines lose all
profanity marking in Spanish (§5). Ranking by enrichment puts `¡Chinga tu madre!` first,
which is what a Mexican speaker says. The corpus knows what subtitlers wrote; enrichment is
what separates that from what is Mexican.

**`data/mx_docs.tsv` is the boundary index**, 241 rows of integers. It carries no
text, so redistributing it raises none of the questions in §1. It is valid on any machine
because OPUS v2024 is a frozen release pinned by sha256 in `subtext fetch`, and `pair_id`
is a deterministic counter in `ingest/parallel.py` — so a range denotes the same lines
everywhere. Without that pinning the index would be meaningless to anyone else.

What it buys is verification rather than time: a reader can check which ranges we claim
are Mexican without re-deriving them or trusting the lexicon. It does not avoid the 4.2 GB
download, because the Spanish text is what the retrieval actually cites.

`demo/` holds 20 cited rows of that retrieval output for readers who will not run anything.

### `mx_embeddings` — 577,035 vectors, 861 MB

Semantic search over the Mexican corpus, so a cue can find precedent even when its exact
wording never occurs.

| | |
|---|---|
| model | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions |
| runs on | the CPU, locally — **no API, no key, no cost** |
| build | `uv run subtext embed-mx`, **85 s** for the whole corpus (6,790 lines/s) |
| stored | one vector per *distinct* English line: 269,869 of 335,800 rows (80.4%) |
| query | HNSW index, **6 ms** — against 41 ms for a full scan, at 100% recall@10 |
| index | `vector_similarity('hnsw','cosineDistance',384)`, 496 MB, 34 s to build |

Deduplicating is worth the join: `What?` occurs 2,140 times and one vector answers for all
of them.

**Why local rather than a hosted embedding API.** At this size the laptop finishes in 85
seconds, so a paid service would buy nothing but a bill and a key to manage. The model
embeds the *English* side, which is what an incoming subtitle is, so an English-only model
is the right tool. If the corpus grew by an order of magnitude this decision should be
revisited; at 577k it is not close.

**The vector index has a silent trap.** `GRANULARITY` defaults to 1, which builds one
HNSW graph per 8,192-row granule rather than one spanning the part. The query then still
succeeds, returns ten plausible-looking rows, and is **wrong** — "Where is my car?" came
back with "- About gay stuff." Nothing errors; the only way to notice is to compare against
`SETTINGS use_skip_indexes=0`. With `GRANULARITY 100000000` the index agrees with the full
scan on 200/200 neighbours across 20 queries and is ~7× faster. `VECTOR_INDEX_GRANULARITY`
in `embeddings.py` pins it and a test guards it.

Worth stating plainly: this was a configuration error on our side, not a ClickHouse defect.
It is recorded because the failure is silent, which is the kind that survives to a demo.

**What it is for.** Phrase lookup (`phrase_index`) is exact and citable but silent when the
wording is new: "That is absolutely ridiculous" has no 3- or 4-word phrase anywhere in the
index. Vector search always answers — which is also its danger, since it cannot promise
anyone ever wrote the thing it found. `find_precedent()` in `retrieval.py` runs the
semantic search and then **groups renderings and reports agreement** (`2/11 translators`),
because a misaligned row is nearly always a lone reading of a line that several other rows
agree on. Ranking by agreement pushes that noise down without having to detect it.

The `min_consensus` lever filters on that share. It is **off by default**: no threshold
here has been tuned against a labelled set, and shipping an untuned filter as though it
were solved would be worse than exposing it honestly.

```
$ uv run subtext precedent "What is up, dude?"
  [0.878] ¿Qué onda, güey?     pair_id 66249542
  [0.878] ¿Qué pedo güey?      pair_id 71753840
```

#### The lexicon is a bulk signal, not a rule about one line

The 66 peninsular words in `ingest/mexican.py` are sound for what they do: score a
**1,000-line window** and compare totals with a 2:1 ratio. In bulk over that much text
`coche` really is 2.6× rarer in Mexican productions.

They are wrong as a test of a single line, and the corpus says so plainly. **59 of the 66
appear in Mexican Spanish**; only 7 never do:

| word | occurrences in `mx_corpus` |
|---|---|
| vale | 924 |
| tío | 450 |
| **coche** | **373** |
| piso | 297 |
| tía | 225 |

Two are *more* common in Mexican productions than in the corpus at large (`vales` 2.03×,
`cazadora` 2.11×). And `vale` is not one word: of its 924 occurrences at least 374 are the
verb *valer* or Mexican idiom — **156 are `me vale`**, as in *me vale madre*.

A register gate built on that list therefore rejected `Eso a mí me vale madre.` and
`Súbete al coche, güey.` Both are real corpus lines. Both are unmistakably Mexican. The
gate was worse than no gate, because the system's promise is that its output is grounded
in attested usage — and it was rejecting attested usage.

`localise.py` now separates them. **`NOT_MEXICAN`** (25 forms) can fail a line and holds
only two kinds of thing: Spain's second-person-plural morphology, which Mexico does not
have at all — the 39 `vosotros` lines in the corpus carry Spain verb forms with them
(*llegáis*, *esperáis*, *hacéis*) and are the documented contamination, not usage — and
words attested at most twice in the Mexican corpus, where an occurrence is likelier to *be* the
0.46% leak than evidence of usage. Everything else is **`WATCH`**: reported to a human,
never fatal.

Borderline Spain-only words (`guay` 9, `gilipollas` 11, `ordenador` 11) are deliberately
left in `WATCH`. A frequency cut alone cannot separate them from homographs that mean
something else entirely in Mexico — `tías` is *aunts*, and `pija` is not "posh" — so the
cautious side is the correct side. Re-derive with `sql/peninsular_rates.sql` after any
rebuild; a passing test is not evidence for adding a word.

#### What it still cannot do

Marker density finds the boundary between Mexican and *non*-Mexican content. Where two
Mexican productions sit adjacent in the corpus there is no density change to find, so they
stay merged: **64 documents of 2,000+ lines hold 35.8% of the corpus**. For grounding
translations in attested Mexican Spanish that costs nothing — every line in them is still
Mexican. It only breaks *per-document* claims. Splitting those needs a different signal;
character names are the obvious candidate, since a name runs through one production and
stops dead at its end.

The old build is kept as `mx_corpus_grid` / `mx_docs_grid` for comparison.

### `imdb_titles`, `imdb_akas`

Straight loads of the IMDb TSVs. Load them **positionally** (`FORMAT TSV` after
stripping the header), not with `TSVWithNames`: that format matches by column *name*,
and IMDb's camelCase headers silently leave every snake_case column empty.

---

## 5. What the corpus shows

All figures from the full 103,637,005-row corpus. Reproduce with the queries in this
section; each returns in about two seconds.

**Register: peninsular Spanish outnumbers Mexican 10 : 1.**

| | Lines |
|---|---|
| Peninsular markers (*joder, coño, gilipollas, hostia, follar, vosotros, capullo*) | 213,741 |
| Mexican markers (*chingar, chinga, pinche, güey, no mames, pendejo, órale*) | 21,334 |

**Profanity: 688,351 lines contain a form of "fuck".** What Spanish does with them:

| Treatment | Lines | Share |
|---|---|---|
| Strong equivalent | 314,097 | 45.6% |
| `maldito` family | 80,868 | 11.7% |
| Euphemism (*diablos, demonios, rayos, caray*) | 27,024 | 3.9% |
| **No profanity marker at all** | 272,525 | **39.6%** |

Whether `maldito` counts as softening is a judgement call, so it is reported as its own
bucket rather than folded into either side.

**False friends.** Mexican markers do not map to *fuck* uniformly:

| Term | Uses | Actually translates |
|---|---|---|
| `pendejo` | 7,065 | **asshole** (2,423) over fuck (1,179) |
| `verga` | 5,169 | **dick** (2,264) |
| `güey` | 4,412 | **dude** (1,041) |
| `chingado` | 957 | **fuck**, 53.6% — the highest rate |
| `órale` | 172 | never *fuck*; not profanity |

The *chingar* family plus adjectival `pinche` are the real functional equivalents.

**Titles: Mexico and Spain get different Spanish titles 64.8% of the time** (11,249 of
17,355 corpus films released in both markets). Excluding Catalan, Galician and Basque
aliases, which `region='ES'` also contains.

### Query patterns

A 15-term × 6-pattern breakdown over 103M rows **times out**. Materialise the matching
subset first, then aggregate — 52,050 rows instead of 103M:

```sql
CREATE OR REPLACE TABLE mex_lines ENGINE=MergeTree ORDER BY tuple() AS
SELECT lowerUTF8(source_text) AS en, lowerUTF8(target_text) AS es
FROM aligned_lines
WHERE match(lowerUTF8(target_text), '\b(chingar|chinga|pinche|güey|pendejo|verga)');
```

Join order matters too: ClickHouse builds its hash table from the **right** side, so
put the small table there. `imdb_akas` (59M rows) on the right exhausts memory; the
43k-row film list on the right runs instantly.

---

## 6. What this pipeline cannot tell you

Three limits, all established by testing rather than assumption. Do not build on any of them.

### Sentence-level film attribution does not work

`corpus_films.start_line` / `n_lines` **should** map every sentence to its film. It
does not. The alignment file implies 105,500,306 Moses lines; the file has
105,482,431 — a 17,875-line discrepancy.

Twelve known films were checked by reading the dialogue at their computed offsets.
**Three matched, nine did not:**

| Film | Line | Result |
|---|---|---|
| The Godfather | 9.4M | ✅ |
| Godfather II | 10.0M | ✅ |
| Alien, Empire Strikes Back, Blade Runner, Back to the Future, Die Hard | 11.4–14.8M | ❌ |
| Goodfellas | 16.0M | ✅ |
| Reservoir Dogs, Pulp Fiction, Shawshank, The Matrix | 16.8–21.8M | ❌ |

The failures are **not monotonic** — Goodfellas at 16M is correct while Alien at 11.4M
is not — so this is not a fixed offset that can be corrected. The likely cause is
OPUS's alternative subtitle uploads, which the README warns "produce a lot of repeated
material".

**Consequence:** you can ask *which* films and series are in the corpus. You cannot ask
what happens to profanity *in a particular film*. Doing so would attach a
confident-looking title that is wrong about three times in four.

The fix, unimplemented, is OPUS's raw XML distribution (~10 GB per language), where
each film is its own file under `<year>/<imdb_id>/` — exact by construction, no line
arithmetic.

### A statistical detector cannot read, and it was wrong about a third of the corpus

The detector keeps a document when a 1000-line window carries ten Mexican markers and
twice as many Mexican as peninsular ones. That rule is sound about *documents* and blind
about *strings*: it cannot tell a Mexican word from a proper name spelled the same way, or
from an ordinary verb form. On 2026-09-07 the documents were read by hand, smallest and
largest first, and five entries in a 99-word lexicon turned out not to be markers at all.

| entry | what it actually matched | documents affected |
|---|---|---|
| `simon` | the **name Simon**. 4,407 occurrences mid-sentence against **14** in lower case. It was the single most frequent "marker" in the whole corpus | 310 of 697 |
| `sepa` | the ordinary subjunctive of *saber* — "no creo que sepa", "nadie sepa quién soy" | 244 of 697 |
| `mande` | the verb *mandar* — "que te mande al estadio" — not the Mexican `¿mande?` | 23 |
| `chin` | the character **Chin Li** | 41 |
| `feria` | the **Science Fair** | 98 |

Two more were removed as judgement calls rather than errors: `lana` is generic before it is
slang (*lana de borrego*), and `huevón` is real in Mexico but equally Colombian and Chilean —
it had pulled in two Colombian productions on its own.

**The lesson generalises: a marker has to be rare *as a string*, not Mexican *as a sense*.**
`simón` = yes is perfectly good Mexican slang and was still the worst entry in the list.

Three defences now exist, and they are in the build rather than in a notebook:

1. **Mid-sentence capitals are stripped before matching** (`normalised()`), so `Chava`
   in *Fiddler on the Roof*, `Morra` in *Limitless* and `Gacha` in *Narcos* stop scoring.
   Sentence-initial capitals are kept: `Ándale,` and `Órale.` open a subtitle line
   constantly and are exactly the usage being hunted. Lines typeset entirely in upper case
   are exempt, or the rule would delete them.
2. **A document needs two *distinct* markers.** One word repeated is evidence about a
   word, not about a film — `pendejo` ×33 and nothing else read as US content once
   somebody looked.
3. **`data/mx_excluded.tsv`** lists twelve documents removed after a human read them, each
   with its reason and its `pair_id` range. Ranges, not document ids: ids are assigned
   during the build and change between runs. The match is by overlap, so a boundary that
   shifts a few lines cannot revive a document a reader rejected. Delete the file and the
   9,150 lines come back — the exclusions are data, not a hand edit to a table.

What this cost, and what it bought, is in §3. The honest summary is that **roughly half the
corpus was resting on strings that were not markers**, and no amount of threshold tuning
would have found it. Somebody had to read the lines.

### `region` is not country of origin

IMDb's `region` is where a title was **released**, not where it was made. Almost every
Hollywood film carries an `MX` row. `region='MX'` means *"this got a Mexican release
under this title"* — useful for the title-divergence analysis, useless as a "this is a
Mexican production" filter.

Nor can origin be inferred from `title.akas.language`: that column is **empty on all
42,963 original-title rows** in this corpus. Country of origin requires an external
source. Wikidata has one, but coverage is uneven and it is openly editable; TMDB's
`production_countries` is better curated.

---

## 7. Attribution

- Subtitle corpus: [OPUS OpenSubtitles v2024](https://opus.nlpl.eu/), from
  [opensubtitles.org](http://www.opensubtitles.org/). Cite Lison & Tiedemann, LREC 2016.
- Title metadata: [IMDb non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/).
  Information courtesy of IMDb. Used under their terms for personal and non-commercial use.
