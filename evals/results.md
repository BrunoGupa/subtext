# Evaluation results

**Corpus:** `the-lighthouse-contract`, 198 lines · 6 seasons · 12 episodes · 6 characters
(`data/sample/`, synthetic and original to this repo).
**Golden set:** 34 hand-labelled questions · 179 labelled line ids · 4 expected-empty
(`evals/golden_set.jsonl`).
**Embeddings:** `all-MiniLM-L6-v2`, 384-dim, local.
**Retrieval:** brute-force `cosineDistance` over `Array(Float32)` in ClickHouse.
**Reproduce:** `make eval` and `make sweep`. Raw output in `results.json` / `sweep.json`.

`k` throughout is a budget of **retrieved lines**, not chunks. That matters: a `window3`
chunk resolves to three lines, so at k=10 the window strategies get roughly a third as many
distinct *places* in the corpus as `line` does. Every comparison below is at equal line cost,
which is the honest way to compare them and also the harsher one for windows.

---

## 1. recall@k — of the lines that answer the question, how many came back

| strategy | k=1 | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| `line` | 0.116 | 0.243 | 0.367 | **0.510** | **0.650** |
| `window3` | 0.086 | 0.253 | 0.352 | 0.517 | 0.607 |
| `window5` | 0.106 | 0.233 | 0.358 | 0.479 | 0.544 |

## 2. hit@k — did *anything* useful come back at all

| strategy | k=1 | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| `line` | 0.500 | 0.667 | 0.733 | 0.800 | 0.833 |
| `window3` | 0.367 | **0.767** | **0.867** | **0.900** | **0.933** |
| `window5` | 0.433 | 0.700 | 0.733 | 0.867 | 0.933 |

**The two tables disagree, and that disagreement is the finding.** `line` wins recall@k;
`window3` wins hit@k by a wide margin (0.933 vs 0.833 at k=20; 0.867 vs 0.733 at k=5).

Windows find the right *scene* far more reliably and then spend the line budget on the
neighbours of the answer instead of on more answers. Single lines are precise but brittle:
when a one-line chunk misses, it misses completely, and 5 of 34 questions are still missed
entirely at k=20.

Which one is "better" depends on what happens downstream, and this is the part worth saying
out loud in an interview: if a language model reads the retrieved context, `window3` is the
better retriever, because the model can use a neighbouring line and cannot use a line that
never arrived. If the retrieved ids are consumed programmatically — counted, joined,
aggregated, as the hybrid path does — `line` is better, because every slot is a distinct
countable event. **This project uses both**, and the counting path deliberately runs on
`line`.

---

## 3. Where each strategy wins, and why

Per-question recall@10, `window3` minus `line`:

| Question | line | window3 | Δ | What the question actually wants |
|---|---|---|---|---|
| q14 "who refuses to lie for Vale" | 0.00 | 1.00 | **+1.00** | a scene |
| q26 "the second permit he hides" | 0.40 | 1.00 | +0.60 | a scene |
| q05 "promises to sign the dive log" | 0.50 | 1.00 | +0.50 | a scene |
| q09 "Teddy's list of broken promises" | 0.50 | 0.88 | +0.38 | a scene |
| q19 "who says they always keep their word" | 0.75 | 0.00 | **−0.75** | four unrelated single lines |
| q21 "the denied insurance claim" | 1.00 | 0.33 | −0.67 | three unrelated single lines |
| q28 "does Rhea mean what she says" | 1.00 | 0.50 | −0.50 | two unrelated single lines |
| q20 "asks for a second contract anyway" | 0.80 | 0.40 | −0.40 | one scene, but a long one |

The rule that falls out: **windows win when the answer is a scene and lose when the answer is
a scattered set of individual lines.** Every large negative delta is a question whose expected
lines sit in different episodes with nothing in common but their meaning; the window pulls in
three lines of local context that are irrelevant to the query and pushes a genuine match off
the end of the budget.

---

## 4. The headline question is the hardest case in the set

The demo question — *"how many times does Vale break a promise?"* — is q03, and its neighbours
q02 ("where is he confronted for going back on his word") and q34 ("which season has the most")
are the same retrieval problem.

| | line | window3 | window5 |
|---|---|---|---|
| q02 recall@10 | 0.00 | 0.09 | — |
| q03 recall@20 | **miss** | **miss** | **miss** |
| q34 recall@10 | 0.00 | 0.20 | — |

**Why it fails.** The corpus never contains the words "breaks a promise". A promise being
broken is written the way people actually write it — as an accusation about something else
entirely:

> *"You said an hour. It did not make forty minutes."* (line 9)
> *"You swore. Eleven hours ago, in front of Rhea, you swore."* (line 49)
> *"The envelope is a crew cut. You said a full share."* (line 101)

Embedded on its own, line 9 is a sentence about a compressor. It is genuinely far in vector
space from "break a promise" — and correctly so. Nothing about the sentence *means*
promise-breaking without the promise it refers to, which was made 5 lines and one scene
earlier. This is why `window3` recovers part of q02 and `line` gets nothing: the window
sometimes catches the promise and the accusation in the same chunk.

**This is a retrieval failure, not a generation failure.** No prompt, no model upgrade and no
instruction about carefulness fixes it, because the evidence is never put in front of the
model. The fixes that would actually work are all retrieval-side, and are listed in §6.

It is also why the agent is not permitted to answer q03 from vector search alone. The hybrid
path (`aggregate_semantic_matches`) retrieves candidates *and* counts them in SQL, and the
count is reported as what it is — the number of semantically matching lines above a distance
threshold, not a fact about a fictional person's conduct.

---

## 4b. A filter bug the demo question exposed

Worth recording because it looked like a retrieval problem and was not.

The obvious way to scope *"how many times does Vale break a promise?"* is
`WHERE character = 'Vale'`. That filter returns almost nothing, and the reason is the same
one as §4: **Vale's broken promises are not in Vale's lines.** They are in Marisol's, Teddy's
and Rhea's — *"You said an hour"*, *"You swore"*, *"You said a full share"*. Scoping to the
speaker throws away the entire evidence base while looking perfectly reasonable in the SQL.

The fix is a second filter, `involving`, matching `character OR speaks_to`, and the agent's
tool descriptions now say explicitly which to reach for. On the demo question, at
`window3`, `max_distance=0.75`, `k=60`:

| filter | matches returned |
|---|---|
| `character = Vale` | 2 |
| `involving = Vale` | 12 |

The general lesson, and the one that transfers off this corpus: **in dialogue data the
subject of a claim and the speaker of the evidence are usually different rows.** Any schema
that only models the speaker will quietly answer "who said this" when the question was "who
did this".

---

## 5. Abstention: the distance threshold does not separate

Four questions in the golden set have no answer in the corpus (Vale's daughter, a sunken cargo
plane, Okonkwo on Vale's drinking, and a purely structural time filter). Brute-force vector
search always returns `k` rows, so refusing requires a distance cut-off. Measured on `line`:

| max_distance | correct abstention (n=4) | false abstention (n=30) |
|---|---|---|
| 0.50 | 0.25 | 0.100 |
| 0.55 | 0.25 | 0.033 |
| 0.60 | 0.00 | 0.033 |
| 0.65 | 0.00 | 0.000 |
| 0.70+ | 0.00 | 0.000 |

**There is no good threshold.** The best available (0.55) catches one unanswerable question in
four while already silencing one answerable question in thirty. The nearest chunk to *"what
does Okonkwo say about Vale's drinking"* sits at a distance indistinguishable from a real
match, because the corpus is full of Okonkwo talking to Vale about Vale's behaviour.

The consequence for the architecture: **abstention cannot be a threshold, so it has to be a
step.** That is what `verify_answer` is for — the agent checks its citations against what was
actually retrieved before it speaks, and the instruction requires it to report an empty result
as an empty result. A confident answer to q31–q33 is the failure mode this catches, and it is
a *generation* failure, which is exactly the kind a prompt-side control can address.

---

## 6. What would move these numbers

In the order the evidence supports, not the order they are fun to build:

1. **A promise-resolution pass.** The genuine fix for q02/q03/q34: for each line that reads as
   an accusation, retrieve the promise it refers to by searching earlier lines from the same
   pair of characters. This is a retrieval-side fix for a retrieval-side failure.
2. **Both strategies indexed at once**, and the strategy chosen per question — windows for
   "what happened in this scene", lines for "how many times". The chunk table already keys on
   `(strategy, window_size)`, so this is a routing decision, not a schema change.
3. **A larger corpus.** 198 lines is small enough that a single well-placed distractor moves
   recall@1 by 3 points. Every number here should be re-run at 10k+ lines before it is quoted
   as anything but a direction. The corpus is a parameter (`--source srt`), not a fixture.
4. **A stronger embedding model.** MiniLM is 384-dim and fast; the failures in §4 are the kind
   a larger model partially absorbs. Worth measuring, not worth assuming.

---

## 7. Answer faithfulness

⏳ **Not yet measured — requires a `GOOGLE_API_KEY`.** The harness is written and runs with
`make eval` plus `--with-agent`; see `reel_query/evaluation.py:evaluate_faithfulness`.

It scores three things per question, all mechanically:

- **citation faithfulness** — of the line ids the answer leans on, the fraction that a
  retrieval tool actually returned in that same run. Catches the fluent answer that cites a
  line the retriever never surfaced.
- **self-verification pass rate** — how often the agent's own `verify_answer` call came back
  clean. Measured independently of the agent's own claim about it.
- **correct abstention on empty** — on q30–q33, whether the agent cited nothing, as it should.

This section gets numbers as soon as the key is in `.env`; it is the last unmeasured clause.
