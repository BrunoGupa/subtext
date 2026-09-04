# Frozen demo — what grounding actually retrieves

This directory exists so the result can be checked **without installing anything**. It is
20 real rows of retrieved evidence, each one a line an actual Mexican translator wrote,
each one traceable back to the corpus by its `pair_id`.

If you would rather run it yourself, skip to [Reproducing it](#reproducing-it).

## The claim being demonstrated

Ask a language model for "Mexican Spanish" and it produces a stereotype. Ask this corpus
and you get what Mexican subtitlers actually wrote, with counts and citations. `evidence-pack.tsv`
is the retrieval output for everyday English phrases: 20 rows covering 17 distinct register markers.

A few rows, to show the shape:

| English | Mexican Spanish | what it shows |
|---|---|---|
| Hi. What's up? | Hola. ¿Qué onda? | neutral Spanish would be *¿Qué pasa?* |
| Your jacket is in the dresser. | Oye, tu **chamarra** está en el vestidor. | Spain says *cazadora* |
| -Cool, brother. | - **Chido**, **carnal**. | two markers in four words |
| Because... I want to talk to you. | Porque quiero **platicar** contigo. | Spain says *hablar* |
| You guys are serious. | **Ustedes** hablan en serio. | never *vosotros* — 59 of those in 718,925 lines |

Note the three rows for *car*: **auto**, **carro** and **coche** all appear. The corpus does
not pretend the choice is uniform, and that honesty is the point — a model asked to sound
Mexican picks one and commits to it.

## Why quoting these lines is legitimate

These are twenty single subtitle lines, none longer than eight words, quoted to demonstrate a
measured linguistic result and individually attributed. They are **not** a redistribution of
the corpus: the corpus itself is 103.6 million pairs and is not in this repository, by
deliberate policy (see `CORPUS.md` §1). The loader downloads it from its original host.

**Source.** OpenSubtitles v2024, English–Spanish, via OPUS.
<http://www.opensubtitles.org/> — the archive's README asks that this link appear in any
report or publication produced with the data, and it does, here and in `CORPUS.md`.

**Citation.** P. Lison and J. Tiedemann, 2016. *OpenSubtitles2016: Extracting Large Parallel
Corpora from Movie and TV Subtitles.* LREC 2016.

**Traceability.** Every row carries its `pair_id`, so any claim here can be checked against
the corpus rather than taken on trust. `doc_id` gives the Mexican document it came from;
those ranges are published in `../data/mx_docs.tsv`.

## Reproducing it

The document ranges are shipped, so the detection does not have to be re-run:

```bash
make up                                    # start ClickHouse
uv run subtext fetch                       # download OPUS (~4.2 GB, checksum-pinned)
uv run subtext load --source parallel      # load 103.6M pairs
uv run subtext build-mx --from-index data/mx_docs.tsv
```

Or derive the boundaries from scratch instead, which adds one full scan:

```bash
uv run subtext build-mx
```

Both produce the same tables — 697 documents, 718,925 lines, 32,074 Mexican markers,
372,575 indexed phrases. The shipped index is valid on any machine because OPUS v2024 is a
frozen release pinned by sha256 in `subtext fetch`, and `pair_id` is a deterministic counter,
so range 66,247,250–66,251,924 means the same lines everywhere.

## What this demo does not show

It is retrieval only — no model was called to produce it. The comparison that matters for
the submission is *grounded output against cold-model output on the same input*, and that
needs an API key and costs money, so it is not frozen here.
