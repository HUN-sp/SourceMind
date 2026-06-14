"""API test — the brief asks for at least one. Hits /health via FastAPI's
TestClient (no LLM call, no Groq tokens)."""
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_endpoint_reports_status():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "indexed_chunks" in body


def test_ask_rejects_empty_question():
    # Pydantic min_length=1 -> 422 before any model call.
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422
