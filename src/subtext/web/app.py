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
import json
import os
import threading
import time
from collections import OrderedDict, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..guard import CueRejected, clean_cue

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


#: Full pipeline runs over the evaluation sets, one page per set, rendered ahead of time
#: so a judge can read 142 lines of output without spending 142 lines of Gemini. Whitelisted
#: by name: a path parameter must never become a file read.
EXAMPLE_PAGES = {
    "paper": "paper-block-40.html",
    "film": "film-block-102.html",
}


@app.get("/examples/{name}", response_class=HTMLResponse)
async def example_page(name: str) -> str:
    file = EXAMPLE_PAGES.get(name)
    path = STATIC / "examples" / file if file else None
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="no such example page")
    return path.read_text(encoding="utf-8")


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
    #: Deliberately far above what `clean_cue` will accept. The request model is the outer
    #: bound on what may be read into memory at all; the contract a person needs explained
    #: -- twelve words, one line -- is enforced below, where the answer can say so. Rejected
    #: here instead, a pasted paragraph gets a bare 422 and the page shows a status code.
    line: str = Field(min_length=1, max_length=4000)


#: Requests one address may make per window, and the window. A person exploring the demo
#: makes a handful; a loop makes thousands, and every one of them is four to six paid
#: Gemini calls. The bound is on cost, not on courtesy.
RATE_LIMIT, RATE_WINDOW = 20, 60.0

#: And a ceiling for the whole day, because a rate limit per address is no defence at all
#: against many addresses. Reaching it takes the site read-only rather than silently
#: spending: the corpus still answers, the model does not.
DAILY_BUDGET = int(os.getenv("SUBTEXT_DAILY_BUDGET", "1500"))

_hits: dict[str, list[float]] = defaultdict(list)
_day: list = [None, 0]        # [date, calls]
_cache: OrderedDict[str, dict] = OrderedDict()
_CACHE_MAX = 256
_guard_lock = threading.Lock()

#: The shape of the payload. Bumped whenever a field is added, so a cached answer from an
#: older build is recomputed rather than rendered with a hole in it.
CACHE_VERSION = 4

#: Answers also live in ClickHouse, keyed by the cleaned line. The in-memory dict above is
#: lost on every deploy and every scale-to-zero, which for a demo means the same five example
#: lines are paid for again each time the site wakes up, and a judge who arrives first waits
#: 20 s where the second one waits none. ClickHouse Cloud is already the shared state of the
#: system, and a ReplacingMergeTree keyed on the line is a cache with no code of its own.
CACHE_DDL = """
CREATE TABLE IF NOT EXISTS web_cache (
    line String, payload String, created DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(created) ORDER BY line
"""
_cache_table_ready = False


def _cache_key(line: str) -> str:
    return f"v{CACHE_VERSION}|{line}"


def _cache_get(line: str) -> dict | None:
    key = _cache_key(line)
    with _guard_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return {**hit, "cached": "memory"}
    try:
        from ..db import client
        rows = client().query(
            "SELECT payload FROM web_cache WHERE line = {k:String} ORDER BY created DESC LIMIT 1",
            parameters={"k": key}).result_rows
    except Exception:  # a cache that cannot be read is a miss, never an error
        return None
    if not rows:
        return None
    answer = json.loads(rows[0][0])
    with _guard_lock:
        _cache[key] = answer
    return {**answer, "cached": "clickhouse"}


def _cache_put(line: str, answer: dict) -> None:
    global _cache_table_ready
    key = _cache_key(line)
    with _guard_lock:
        _cache[key] = answer
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    try:
        from ..db import client
        ch = client()
        if not _cache_table_ready:
            ch.command(CACHE_DDL)
            _cache_table_ready = True
        ch.insert("web_cache", [[key, json.dumps(answer, ensure_ascii=False)]],
                  column_names=["line", "payload"])
    except Exception:  # the answer is already on its way to the person; the cache can wait
        pass


#: Google Translate, as the second control. It has no Mexican Spanish to ask for -- the
#: target is `es`, one Spanish -- and on the evaluation lines it answers in peninsular:
#: `coche`, `chaqueta`, `conduce`, `coño`. That is the product's argument made by a
#: neighbour, so it is shown, uncited, under the readings. Reached with the runtime's own
#: credentials (the Cloud Run service account, or gcloud locally); absent those, or the API,
#: it is simply not shown. Free below 500,000 characters a month.
TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
_google_creds = None


def _google_token() -> tuple[str, str | None]:
    """A bearer token and the project to bill. Application default credentials first --
    that is the Cloud Run service account -- and, on a developer machine where those may be
    stale, the gcloud CLI's own token."""
    global _google_creds
    import google.auth
    import google.auth.transport.requests

    try:
        if _google_creds is None:
            _google_creds, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-translation"])
            _google_creds._quota_project = os.getenv("GOOGLE_CLOUD_PROJECT") or project
        if not _google_creds.valid:
            _google_creds.refresh(google.auth.transport.requests.Request())
        return _google_creds.token, _google_creds._quota_project
    except Exception:
        import subprocess
        token = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                               text=True, timeout=15, check=True).stdout.strip()
        return token, os.getenv("GOOGLE_CLOUD_PROJECT")


def _google(line: str) -> dict[str, Any]:
    from ..localise import check_register

    try:
        import requests

        token, project = _google_token()
        headers = {"Authorization": f"Bearer {token}"}
        if project:
            headers["x-goog-user-project"] = project
        r = requests.post(TRANSLATE_URL, headers=headers, timeout=10,
                          json={"q": [line], "source": "en", "target": "es", "format": "text"})
        r.raise_for_status()
        spanish = r.json()["data"]["translations"][0]["translatedText"].strip()
    except Exception as exc:
        return {"spanish": "", "error": str(exc)[:120], "not_mexican": [], "watch": []}
    return {"spanish": spanish, "error": "", **_register(spanish)}


def _rate_ok(who: str) -> bool:
    now = time.time()
    with _guard_lock:
        seen = [t for t in _hits[who] if now - t < RATE_WINDOW]
        _hits[who] = seen
        if len(seen) >= RATE_LIMIT:
            return False
        seen.append(now)
        # Addresses that stopped asking should not be remembered forever.
        if len(_hits) > 4096:
            for key in [k for k, v in _hits.items() if not v]:
                del _hits[key]
        return True


def _budget_ok() -> bool:
    today = date.today()
    with _guard_lock:
        if _day[0] != today:
            _day[0], _day[1] = today, 0
        if _day[1] >= DAILY_BUDGET:
            return False
        _day[1] += 1
        return True


def _asker(model: str | None = None):
    """The project's Gemini caller -- the same one the CLI uses.

    This used to be a second, thinner copy that never produced a `BLOCKED` reason, so the
    path a judge exercises reported every provider block as "no reason given".
    """
    from ..gemini import asker
    return asker(model)


def _register(spanish: str) -> dict[str, list[str]]:
    """The same lexicon the readings are gated on, applied to a control. `not_mexican` is
    what fails a reading; `watch` is Spain's usual word where Mexican subtitlers also use
    it sometimes -- `joder`, `coño` -- reported, never failed, and worth seeing on a control
    because it is exactly the register the controls drift into."""
    from ..localise import check_register

    report = check_register(spanish)
    return {"not_mexican": list(report.not_mexican) + list(report.other_latam),
            "watch": [w for w in report.watch if w not in DEFENDED]}


#: Watch-list words the README explicitly defends as Mexican usage too -- `coche` 373 times
#: in the Mexican corpus, `vale` 924 -- so flagging them on a control would contradict the
#: project's own finding on the same page.
DEFENDED = frozenset({"coche", "vale"})


BASELINE_PROMPT = ("Translate this English subtitle line into Mexican Spanish. "
                   "Reply with the Spanish line only, nothing else.\n\n")


def _baseline(line: str, ask) -> dict[str, Any]:
    from ..localise import check_register
    from ..variants import BLOCKED

    try:
        reply = (ask(BASELINE_PROMPT + line) or "").strip()
    except Exception as exc:  # the control failing must not take the answer down with it
        return {"spanish": "", "error": str(exc)[:120], "not_mexican": []}
    if not reply or reply.startswith(BLOCKED):
        return {"spanish": "", "error": reply or "empty response", "not_mexican": []}
    spanish = reply.splitlines()[0].strip().strip('"“”')
    return {"spanish": spanish, "error": "", **_register(spanish)}


def _localise(line: str) -> dict[str, Any]:
    from ..address_llm import cached_tagger
    from ..variants import translate_variants

    from ..localise import gather_phrases

    model = settings().gemini_model
    ask = _asker(model)
    started = time.perf_counter()
    # Form-neutral and fetched once for the cue, so it rides on the response rather than
    # on every reading: what a phrase agrees on is the same for tú, usted and ustedes.
    # Once for the whole request, too: `translate_variants` needs the identical object to
    # build its prompts, and fetching it here and again in there ran the phrase channel
    # twice -- 20 round trips to ClickHouse Cloud where 10 answer the question.
    # The phrase channel, the vector channel and the two controls need nothing from each
    # other, so they start together. Measured before this: 16-28 s per new line, most of
    # it four things waiting in a row for the same network.
    from concurrent.futures import ThreadPoolExecutor
    from ..variants import group_by_address

    tag_forms = cached_tagger(ask, model=model)

    def timed_phrases():
        # The number the page reports as ClickHouse Cloud is this channel alone: ten SQL
        # queries and nothing else. The vector channel beside it embeds and tags with
        # Gemini, and counting that as the database would flatter the wrong thing.
        t = time.perf_counter()
        return gather_phrases(line), time.perf_counter() - t

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_phrases = pool.submit(timed_phrases)
        f_readings = pool.submit(group_by_address, line, tag_forms=tag_forms)
        f_baseline = pool.submit(_baseline, line, ask)
        f_google = pool.submit(_google, line)
        phrases, corpus_seconds = f_phrases.result()
        readings = f_readings.result()
        variants = translate_variants(line, ask=ask, phrases=phrases, readings=readings,
                                      tag_forms=tag_forms)
        baseline, google = f_baseline.result(), f_google.result()
    total_seconds = time.perf_counter() - started
    return {
        # The control: the same model asked the same thing with no evidence at all. It is
        # what a person gets today, and the difference between it and the readings above
        # is the whole product. Never gated, never retried, never cited -- it is shown as
        # what it is, and run through the register lexicon so its peninsular words show.
        "baseline": baseline,
        "google": google,
        "line": line,
        # Where the time went, so the page can say it. The phrase channel is ten round
        # trips to ClickHouse Cloud; everything after it is the vector channel plus one
        # Gemini call per reading. A judge who sees "14 s" deserves to know which half.
        "timing": {"corpus_s": round(corpus_seconds, 2), "total_s": round(total_seconds, 1)},
        "cached": False,
        "agreed": [
            {"phrase": hit.phrase, "support": hit.support, "spanish": c.spanish,
             "lines": c.lines, "of_lines": c.of_lines, "enrichment": round(c.enrichment, 1)}
            for hit in phrases for c in hit.consensus
        ],
        # One reading means the English settled the person; several mean it left it open,
        # and only then is there a choice to offer.
        "ambiguous": len(variants) > 1,
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
                # Whether the answer addressed the person it was asked to. Coming back
                # UNMARKED is not a failure -- Spanish drops the subject pronoun -- so this
                # is false only when the model addressed somebody else, which a reviewer
                # needs to see and a reader of the page could not otherwise tell.
                "form_confirmed": v.form_confirmed,
                "not_mexican": v.not_mexican,
                # A reading the model did not answer is not a refusal and must not be
                # rendered as one -- see `variants.BLOCKED`.
                "answered": v.answered,
                "block_reason": v.block_reason,
                "gender": v.gender.value if v.gender else None,
                "other_gender": v.other_gender,
                "agreed": list(v.agreed),
                # Quotations, shown as their translator wrote them. Never re-gendered:
                # a `pair_id` that points at a line nobody wrote is worth less than none.
                "alternatives": [
                    {"spanish": a.spanish, "english": a.english, "pair_id": a.pair_id,
                     "similarity": round(a.similarity, 2), "times": a.times,
                     "of": a.english_times}
                    for a in v.alternatives
                ],
            }
            for v in variants
        ],
    }


@app.post("/api/localise")
async def api_localise(request: LocaliseRequest, http: Request) -> dict[str, Any]:
    """One English subtitle line in, every Mexican reading the corpus attests out."""
    try:
        line = clean_cue(request.line)
    except CueRejected as why:
        return {"line": request.line, "readings": [], "error": str(why)}

    if not settings().has_gemini_key:
        return {"line": line, "readings": [],
                "error": "GOOGLE_API_KEY is not set, so no translation can be written."}

    # Repeats cost nothing and the demo repeats a lot: five example buttons, one judge,
    # one video take, the same five lines each time. Said out loud in the payload: a repeat
    # answer is instant because it is a repeat, not because the corpus is fast, and the
    # page must not advertise one as the other.
    cached = await asyncio.to_thread(_cache_get, line)
    if cached is not None:
        return cached

    who = (http.client.host if http.client else "?")
    if not _rate_ok(who):
        return {"line": line, "readings": [],
                "error": f"Too many requests — {RATE_LIMIT} a minute. Try again shortly."}
    if not _budget_ok():
        return {"line": line, "readings": [],
                "error": "The day's translation budget is spent. The corpus is still "
                         "searchable; come back tomorrow for a new line."}

    answer = await asyncio.to_thread(_localise, line)
    await asyncio.to_thread(_cache_put, line, answer)
    return answer


@app.post("/api/evidence")
async def api_evidence(request: LocaliseRequest, http: Request) -> dict[str, Any]:
    """What the corpus holds for this line, with no model call and no translation.

    No model runs here, so nothing is spent on Gemini -- which is exactly why this endpoint
    was left open, and exactly why that was wrong. `LocaliseRequest` admits 4000 characters:
    ~700 words become ~2,700 n-grams in one `IN (...)` against ClickHouse Cloud, which bills
    for the compute. Guarding the paid door and leaving this one unbounded guards nothing.
    """
    from ..localise import gather_phrases

    try:
        line = clean_cue(request.line)
    except CueRejected as why:
        return {"line": request.line, "phrases": [], "error": str(why)}

    who = (http.client.host if http.client else "?")
    if not _rate_ok(who):
        return {"line": line, "phrases": [],
                "error": f"Too many requests — {RATE_LIMIT} a minute. Try again shortly."}

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
            for h in gather_phrases(line)
        ]

    return {"line": line, "phrases": await asyncio.to_thread(work)}


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run("subtext.web.app:app", host=host, port=port, reload=reload)
