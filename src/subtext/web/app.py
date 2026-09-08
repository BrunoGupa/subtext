"""The web front end.

The hackathon rules require a submission to "run on at least one of the following
platforms: web, Android, or iOS", so this is not polish — it is the deliverable that
makes the project submittable at all.

It is also the demo. The interesting thing about this project is not that it returns an
answer; it is *how* it gets there: which tools it chose, the SQL it actually ran, the
dialogue it retrieved, and whether its own check of that answer passed. So the UI shows
all of it by default rather than hiding it behind a disclosure triangle.

Two modes, deliberately:

* **Retrieval** — vector search and the hybrid aggregation. No model, no API key, works
  on a clean clone. This is the half that the evaluation numbers describe.
* **Agent** — the full ADK loop. Needs `GOOGLE_API_KEY`; degrades to a clear message
  rather than an error page when there isn't one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..config import settings

STATIC = Path(__file__).parent / "static"

# One agent run at a time. The demo is public and each run costs Gemini tokens and a
# subprocess; queueing is friendlier than falling over, and slow is better than broken.
_agent_lock = asyncio.Semaphore(1)

app = FastAPI(
    title="Subtext",
    description="Hybrid retrieval (vector + SQL) over a film dialogue corpus in ClickHouse.",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=50)
    strategy: Literal["line", "window"] = "line"
    window_size: int = Field(default=1, ge=1, le=9)
    character: str | None = Field(default=None, max_length=60)
    involving: str | None = Field(default=None, max_length=60)
    season: int | None = Field(default=None, ge=0, le=99)


class AggregateRequest(SearchRequest):
    k: int = Field(default=60, ge=1, le=200)
    group_by: str = Field(default="season", max_length=80)
    max_distance: float = Field(default=0.75, ge=0.0, le=2.0)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    from ..agent import clickhouse_mcp
    from ..ingest.load import loaded_strategies
    from ..retrieval import corpus_stats

    s = settings()
    try:
        stats = corpus_stats()
        chunks = [
            {"strategy": st, "window_size": w, "chunks": n} for st, w, n in loaded_strategies()
        ]
        connected = True
    except Exception as exc:  # the UI should say "database down", not show a stack trace
        stats, chunks, connected = {"error": str(exc)}, [], False

    return {
        "clickhouse_connected": connected,
        "corpus": stats,
        "chunk_tables": chunks,
        "embedding_model": s.embedding_model,
        "gemini_model": s.gemini_model,
        "agent_available": s.has_gemini_key,
        "mcp_available": clickhouse_mcp.is_available(),
    }


class LocaliseRequest(BaseModel):
    line: str = Field(min_length=1, max_length=300)


def _asker(model: str | None = None):
    from google import genai
    from google.genai import types

    client = genai.Client()
    name = model or settings().gemini_model
    config = types.GenerateContentConfig(
        temperature=0.2, max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    def ask(prompt: str) -> str:
        return client.models.generate_content(
            model=name, contents=prompt, config=config).text or ""

    return ask


def _localise(line: str) -> dict[str, Any]:
    from ..address_llm import cached_tagger
    from ..variants import translate_variants

    model = settings().gemini_model
    ask = _asker(model)
    variants = translate_variants(line, ask=ask, tag_forms=cached_tagger(ask, model=model))
    return {
        "line": line,
        "readings": [
            {
                "form": v.form.value,
                "spanish": v.spanish,
                "pair_ids": list(v.pair_ids[:4]),
                "register": v.register,
                "grounded": v.grounded,
                "weak": v.weakly_grounded,
                "similarity": round(v.top_similarity, 2),
                "well_formed": v.well_formed,
                "not_mexican": v.not_mexican,
            }
            for v in variants
        ],
    }


@app.post("/api/localise")
async def api_localise(request: LocaliseRequest) -> dict[str, Any]:
    """One English subtitle line in, every Mexican reading the corpus attests out."""
    if not settings().has_gemini_key:
        return {"line": request.line, "readings": [],
                "error": "GOOGLE_API_KEY is not set, so no translation can be written."}
    return await asyncio.to_thread(_localise, request.line)


@app.post("/api/evidence")
async def api_evidence(request: LocaliseRequest) -> dict[str, Any]:
    """What the corpus holds for this line, with no model call and no translation."""
    from ..localise import gather_phrases

    def work():
        return [
            {
                "phrase": h.phrase,
                "lines": h.support,
                "renderings": [
                    {"spanish": r.spanish, "count": r.count,
                     "pair_id": r.pair_id, "english": r.english,
                     "coverage": r.coverage}
                    for r in h.renderings
                ],
            }
            for h in gather_phrases(request.line)
        ]

    return {"line": request.line, "phrases": await asyncio.to_thread(work)}


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run("subtext.web.app:app", host=host, port=port, reload=reload)
