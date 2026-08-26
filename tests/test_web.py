"""Web layer tests.

These deliberately do not touch ClickHouse or Gemini: they check the contract the
browser depends on and the guards that matter once the app is publicly reachable.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from subtext.web.app import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_the_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Subtext" in response.text
    assert "text/html" in response.headers["content-type"]


@pytest.mark.parametrize(
    "body",
    [
        {},                                   # no question
        {"question": ""},                     # empty question
        {"question": "a" * 501},              # over the length cap
        {"question": "x", "k": 0},            # k below range
        {"question": "x", "k": 999},          # k above range
        {"question": "x", "strategy": "hnsw"},  # not a real strategy
        {"question": "x", "season": -1},      # negative season
    ],
)
def test_search_rejects_bad_input_before_touching_the_database(body):
    assert client.post("/api/search", json=body).status_code == 422


def test_aggregate_rejects_a_group_by_that_is_not_an_allowed_column():
    response = client.post(
        "/api/aggregate",
        json={"question": "x", "group_by": "season; DROP TABLE lines"},
    )
    assert response.status_code == 400
    assert "cannot group by" in response.json()["detail"]


def test_ask_without_a_key_explains_itself_rather_than_erroring(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from subtext.config import settings

    settings.cache_clear()
    try:
        response = client.post("/api/ask", json={"question": "hello"})
        assert response.status_code == 503
        assert "GOOGLE_API_KEY" in response.json()["detail"]
    finally:
        settings.cache_clear()
