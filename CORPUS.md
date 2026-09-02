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

Two limits, both established by testing rather than assumption. Do not build on either.

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
