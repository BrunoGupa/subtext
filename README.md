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
filters them by season, character and timecode. That is the whole architecture, and it is
load-bearing rather than bolted on.

## Stack
- **Gemini + Google ADK** — planning, SQL generation, self-validation (hackathon requirement #1)
- **ClickHouse** — the corpus, the aggregation, *and* the vector search (partner product, requirement #2)
- **sentence-transformers** — local embeddings, $0
- **Python 3.12 + uv**

## Status
🔒 **PRIVATE while under construction.** Goes public before submission — the hackathon requires a
public repo under an OSI licence (Apache-2.0, already in `LICENSE`) and a publicly hosted URL.

## Docs
| File | What |
|---|---|
