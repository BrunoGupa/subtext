# Evaluation

## The previous golden set was withdrawn — 2026-09-07

`golden_set.jsonl`, `results.md`, `results.json` and `sweep.json` were removed, along with the
synthetic `data/sample/` corpus they scored.

The reason is not that the numbers were bad. It is that they could not mean anything. The
corpus was written for this repository by a model, the "hand-labelled" line ids were planted
by the same process that wrote the dialogue, and no human ever reviewed or approved them. A
retriever scored against labels derived from its own source text measures self-consistency.
Calling that *recall* overstated it, and the README said so on the front page.

They are recoverable from git history if a reason to restore them ever appears.

## What replaces it

Evaluation moves to text this project did not write: the Spanish side of a human-translated
subtitle corpus, used as a **reference translation** rather than as a label. A reference is one
valid translation, not the only one, and some references in the corpus are themselves poor or
misaligned — so the reference is scored by a human reviewer, not by string equality.

The evaluation set is the **AFI 100** film quotes, split by what reference exists for each:

| sector | n | reference available |
|---|---|---|
| A | 14 | Mexican Spanish, from the Mexican sub-corpus |
| B | 51 | Spanish of unverified register, from the full corpus |
| C | 35 | none |

Sector A is the only one with a Mexican reference, and several of its 14 rows are misaligned
corpus rows rather than translations of the quote. They are reviewed one by one.

## Metrics, and why there are several

- **recall@k** — of the expected lines, how many came back in the top k. Retrieval quality.
- **hit@k** — did *at least one* expected line come back. A different question.
- **abstention curve** — at each distance cut-off, correct abstention on unanswerable
  questions against false abstention on answerable ones.
- **citation faithfulness** — of the lines the final answer relies on, how many were actually
  retrieved. Needs `GOOGLE_API_KEY`; run `subtext eval --with-agent`.

The split exists to separate **retrieval failure** (the answer never came back — recall@k) from
**generation failure** (it came back and the answer ignored it, or cited something that never
arrived — faithfulness). They have different fixes, and one blended "accuracy" number hides
which one you have.

The code in `src/subtext/evaluation.py` still implements these. It has no data to run against
until the replacement set is built.
