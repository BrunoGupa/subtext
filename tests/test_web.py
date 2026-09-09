"""The web app: one input, every reading the corpus attests.

The endpoints these tests used to cover -- `/api/search`, `/api/aggregate`, `/api/ask` --
served the question-answering product this repo no longer builds. They went with it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from subtext.web.app import app

client = TestClient(app)


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_index_serves_the_page():
    page = client.get("/")
    assert page.status_code == 200
    assert "Subtext" in page.text
    assert "/api/localise" in page.text


def test_the_page_offers_examples_the_corpus_has_never_seen():
    """The first thing a visitor sees has to show what the tool is *for*: English `you` is
    three things at once, and the reading has to be assembled from precedent. `Get in the
    car.` used to be the example, and it is in the corpus four times -- so the page was
    demonstrating a lookup. Decided 2026-09-09: every example is a line the corpus does
    not contain (checked with `SELECT count() FROM mx_corpus WHERE en = ...`, all zero),
    and the page says so beside them."""
    page = client.get("/").text
    for line in ("Come with me if you want to live.", "You talking to me?",
                 "I like your jacket.", "Shut up and drive.", "Get out of my car."):
        assert line in page
    assert "None of these lines exists in the corpus" in page


def test_the_evaluation_pages_are_served_by_name_only():
    """The rendered evaluation pages are whitelisted by name: a path must never become a
    file read. The paper page ships; the film page may not exist yet and must 404 cleanly."""
    assert client.get("/examples/paper").status_code == 200
    assert client.get("/examples/../index").status_code in (404, 400)
    assert client.get("/examples/app.py").status_code == 404


@pytest.mark.parametrize("body", [
    {},                      # nothing at all
    {"line": ""},            # empty
    {"line": "x" * 5000},    # past what the request model will even carry
])
def test_localise_rejects_malformed_requests(body):
    """Validation runs before anything is retrieved or any token is spent."""
    assert client.post("/api/localise", json=body).status_code == 422


@pytest.mark.parametrize("line,tell", [
    ("x" * 350, "characters"),
    (" ".join(["hi"] * 30), "words"),
    ("first line\nsecond line", "One line at a time"),
])
def test_a_line_that_breaks_the_contract_comes_back_explained(line, tell):
    """Answered rather than 422'd: the page renders `error`, and a person who pasted a
    paragraph should read what to do instead of `the server answered 422`. No retrieval
    happens and no token is spent either way."""
    body = client.post("/api/localise", json={"line": line}).json()
    assert body["readings"] == []
    assert tell in body["error"]


def test_localise_without_a_key_explains_itself_rather_than_erroring(monkeypatch):
    """A visitor with no Gemini key should be told what is missing, not shown a stack
    trace. Retrieval still works without one; only writing the Spanish needs a model."""
    import subtext.config as config

    config.settings.cache_clear()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        body = client.post("/api/localise", json={"line": "Shut up!"}).json()
        assert body["readings"] == []
        assert "GOOGLE_API_KEY" in body["error"]
    finally:
        config.settings.cache_clear()


def test_evidence_rejects_bad_input_too():
    assert client.post("/api/evidence", json={"line": ""}).status_code == 422


def test_evidence_applies_the_same_cue_rules_as_the_paid_path():
    """No model runs behind `/api/evidence`, but ClickHouse Cloud does, and it is billed.
    A paragraph here is ~2,700 n-grams in one query."""
    body = client.post("/api/evidence", json={"line": "word " * 400}).json()
    assert body["phrases"] == []
    assert "120" in body["error"]


# --- what keeps the key from being emptied overnight -------------------------------

def test_one_address_is_cut_off_after_its_share():
    """Twenty a minute is generous for a person and useless for a loop. Tested here
    rather than over HTTP: a real request takes about ten seconds, so twenty of them
    outlast the window and nothing would ever trip."""
    # Imported by name: `subtext/web/__init__.py` re-exports the FastAPI instance as
    # `app`, so `import subtext.web.app` hands back the instance, not the module.
    from subtext.web.app import RATE_LIMIT, _hits, _rate_ok

    _hits.clear()
    assert all(_rate_ok("10.0.0.1") for _ in range(RATE_LIMIT))
    assert not _rate_ok("10.0.0.1")
    # A limit per address is no defence against many addresses -- that is the budget's job.
    assert _rate_ok("10.0.0.2")


def test_the_day_has_a_ceiling_and_it_degrades_rather_than_spends():
    from subtext.web.app import DAILY_BUDGET, _budget_ok, _day

    _day[0], _day[1] = None, 0
    for _ in range(DAILY_BUDGET):
        assert _budget_ok()
    assert not _budget_ok()


def test_the_cache_evicts_the_oldest_and_keeps_what_is_asked_for_again():
    """The demo repeats: four example buttons, one judge, one video take. Bounded, so a
    long session cannot grow it without limit."""
    from subtext.web.app import _CACHE_MAX, _cache

    _cache.clear()
    for i in range(_CACHE_MAX + 10):
        _cache[f"line {i}"] = {"readings": []}
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    assert len(_cache) == _CACHE_MAX
    assert "line 0" not in _cache
    assert f"line {_CACHE_MAX + 9}" in _cache


# --- the caller Gemini answers through ---------------------------------------------

def test_a_transient_failure_is_retried_and_a_real_one_is_not():
    """Two of the 102 film-set lines came back 503 UNAVAILABLE, which is 2% of a run
    reaching the page as an error for a reason that is not ours and does not persist.
    Everything else must fail at once: retrying a bad key hides a fault behind a delay."""
    from subtext.gemini import is_transient

    assert is_transient(RuntimeError("503 UNAVAILABLE. The service is currently unavailable."))
    assert is_transient(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not is_transient(RuntimeError("400 INVALID_ARGUMENT: API key not valid"))
    assert not is_transient(RuntimeError("403 PERMISSION_DENIED"))
