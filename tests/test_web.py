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


def test_the_page_offers_an_example_the_corpus_answers_three_ways():
    """The first thing a visitor sees has to show what the tool is *for*, and what it is
    for is that English `you` is three things at once. `Get in the car.` is in the corpus
    as `Súbete al coche.`, `Súbase.` and `- Entren al carro.`"""
    assert "Get in the car." in client.get("/").text


@pytest.mark.parametrize("body", [
    {},                      # nothing at all
    {"line": ""},            # empty
    {"line": "x" * 400},     # past the length bound
])
def test_localise_rejects_bad_input_before_touching_the_model(body):
    """Validation runs before anything is retrieved or any token is spent."""
    assert client.post("/api/localise", json=body).status_code == 422


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
