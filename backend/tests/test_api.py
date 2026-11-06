"""API tests — hit the FastAPI app via TestClient. No network, no Groq tokens:
the index is a tiny synthetic one and the LLM is monkeypatched where needed."""
import json

from fastapi.testclient import TestClient

import app as app_module
from app import app

client = TestClient(app)


def _fake_llm(payload):
    return lambda system, user: json.dumps(payload)


def test_health_endpoint_reports_status(tiny_index):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0
    assert "model" in body


def test_health_flags_a_missing_index(no_index):
    body = client.get("/health").json()
    assert body["status"] == "index_missing"
    assert body["indexed_chunks"] == 0


def test_ask_rejects_empty_question():
    # Pydantic min_length=1 -> 422 before any model call.
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_without_index_is_a_clear_503_not_a_500(no_index):
    resp = client.post("/ask", json={"question": "What is the gross NPA ratio?"})
    assert resp.status_code == 503
    assert "ingest.py" in resp.json()["detail"]


def test_ask_happy_path_returns_answer_and_sources(tiny_index, monkeypatch):
    from rag import pipeline

    fake = _fake_llm({"not_covered": False, "answer": "1.15%", "used_passages": [1]})
    monkeypatch.setattr(app_module, "ask", lambda q, doc=None: pipeline.ask(q, doc=doc, generate=fake))
    resp = client.post("/ask", json={"question": "What is the gross NPA ratio to gross advances?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert "1.15%" in body["answer"]
    assert body["sources"] and body["sources"][0]["page"] >= 1
    assert body["sources"][0]["fiscal_period"]  # exposed so the UI can show the period
