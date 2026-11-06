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


# --- E3: readiness reporting ------------------------------------------------
def test_health_reports_ready_when_index_and_key_present(tiny_index, monkeypatch):
    monkeypatch.setattr(app_module, "llm_configured", lambda: True)
    body = client.get("/health").json()
    assert body["ready"] is True and body["problems"] == []
    assert body["llm_configured"] is True


def test_health_lists_each_setup_problem(no_index, monkeypatch):
    monkeypatch.setattr(app_module, "llm_configured", lambda: False)
    body = client.get("/health").json()
    assert body["ready"] is False and body["llm_configured"] is False
    text = " ".join(body["problems"])
    assert "ingest.py" in text and "GROQ_API_KEY" in text


def test_health_never_leaks_the_api_key(tiny_index, monkeypatch):
    import rag.generate as gen

    monkeypatch.setattr(gen, "CONFIG", type("C", (), {"groq_api_key": "gsk_FAKE_SECRET_VALUE", "llm_model": "m"})())
    assert gen.llm_configured() is True
    assert "gsk_FAKE_SECRET_VALUE" not in client.get("/health").text


def test_llm_configured_rejects_blank_and_placeholder_keys(monkeypatch):
    import rag.generate as gen

    for key, expected in [("", False), ("gsk_your_free_key_here", False), ("gsk_real_looking", True)]:
        monkeypatch.setattr(gen, "CONFIG", type("C", (), {"groq_api_key": key})())
        assert gen.llm_configured() is expected
