# `the-lighthouse-contract` — a synthetic sample corpus

Six seasons of an invented series about a deep-water salvage crew, written for this repo so
that anyone can clone it and get a working end-to-end demo with **no downloads, no accounts
and no licensing questions**. Every line is original to this project and released under the
repo's Apache-2.0 licence.

It exists because the headline question — *"how many times across six seasons does this
character break a promise?"* — needs a corpus with **seasons, named characters, and promises
that are actually broken later on**. Promise-breaks are planted deliberately so the golden
set in `evals/golden_set.jsonl` has real ground truth, alongside promises that are *kept*
(distractors that a naive keyword search will happily return and a good retriever will rank
lower).

scale. For real scale, load your own subtitles instead:

```bash
uv run subtext load --source srt --path /path/to/srt/dir --title "Some Show" --title-id tt1234567
```

## Shape

| | |
|---|---|
| Series | *The Lighthouse Contract* (`tt_lighthouse`) |
| Seasons | 6 |
| Episodes | 2 per season |
| Characters | Vale, Marisol, Okonkwo, Teddy, Rhea, Bosun |

`Vale` is the promise-breaker the demo question is about. `Okonkwo` keeps almost every promise
he makes — he is the control.
