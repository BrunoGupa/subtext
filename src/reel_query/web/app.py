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
    title="Reel Query",
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


@app.post("/api/search")
async def api_search(request: SearchRequest) -> dict[str, Any]:
    from ..retrieval import search

    hits = await asyncio.to_thread(
        search,
        request.question,
        k=request.k,
        strategy=request.strategy,
        window_size=request.window_size if request.strategy == "window" else 1,
        character=request.character or None,
        involving=request.involving or None,
        season=request.season,
    )
    return {
        "count": len(hits),
        "hits": [
            {
                "line_id": h.line_id,
                "season": h.season,
                "episode": h.episode,
                "character": h.character,
                "timecode": h.timecode,
                "text": h.text,
                "distance": round(h.distance, 4),
                "similarity": round(h.similarity, 4),
            }
            for h in hits
        ],
    }


@app.post("/api/aggregate")
async def api_aggregate(request: AggregateRequest) -> dict[str, Any]:
    from ..retrieval import hybrid_aggregate

    columns = tuple(c.strip() for c in request.group_by.split(",") if c.strip())
    try:
        rows, sql = await asyncio.to_thread(
            hybrid_aggregate,
            request.question,
            group_by=columns or ("season",),
            k=request.k,
            strategy=request.strategy,
            window_size=request.window_size if request.strategy == "window" else 1,
            character=request.character or None,
            involving=request.involving or None,
            season=request.season,
            max_distance=request.max_distance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "rows": [
            {
                **{c: r[c] for c in (columns or ("season",))},
                "matches": int(r["matches"]),
                "best_distance": float(r["best_distance"]),
                "line_ids": [int(i) for i in r["line_ids"]],
                "examples": list(r["examples"]),
            }
            for r in rows
        ],
        "total_matches": sum(int(r["matches"]) for r in rows),
        "sql": sql,
    }


@app.post("/api/ask")
async def api_ask(request: AskRequest) -> dict[str, Any]:
    if not settings().has_gemini_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "The agent needs a GOOGLE_API_KEY in .env (free tier at "
                "https://aistudio.google.com/apikey). The retrieval tab works without one."
            ),
        )

    from ..agent import ask_async

    async with _agent_lock:
        try:
            answer = await ask_async(request.question)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    return {
        "answer": answer.text,
        "tool_calls": answer.tool_calls,
        "tools_used": answer.tools_used,
        "retrieved_line_ids": answer.retrieved_line_ids,
        "verified": answer.verified,
    }


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run("reel_query.web.app:app", host=host, port=port, reload=reload)
