"""E1: GET /documents lists what's indexed; /ask rejects an unknown document."""
from fastapi.testclient import TestClient

import app as app_module
import rag.store as store
from app import app

client = TestClient(app)


def test_list_documents_aggregates_index_metadata(tiny_index):
    docs = store.list_documents()
    assert [d["doc"] for d in docs] == ["alpha.pdf", "beta.pdf"]  # sorted by name
    alpha = docs[0]
    assert alpha["pages"] == 2 and alpha["chunks"] >= 2
    assert alpha["fiscal_period"] == "March 31, 2026"
    assert alpha["doc_label"] == "Standalone Results — March 31, 2026"


def test_documents_endpoint(tiny_index):
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert [d["doc"] for d in resp.json()["documents"]] == ["alpha.pdf", "beta.pdf"]


def test_documents_endpoint_without_index_is_empty_not_an_error(no_index):
    assert client.get("/documents").json() == {"documents": []}


def test_ask_unknown_document_is_404_and_skips_the_llm(tiny_index, monkeypatch):
    monkeypatch.setattr(app_module, "limiter", app_module.SlidingWindowLimiter(limit=0))
    called = []
    monkeypatch.setattr(app_module, "ask", lambda *a, **k: called.append(1))
    resp = client.post("/ask", json={"question": "gross npa", "doc": "nope.pdf"})
    assert resp.status_code == 404
    assert "nope.pdf" in resp.json()["detail"]
    assert not called


def test_ask_with_known_document_passes_it_through(tiny_index, monkeypatch):
    monkeypatch.setattr(app_module, "limiter", app_module.SlidingWindowLimiter(limit=0))
    seen = {}

    def fake_ask(q, doc=None):
        seen["doc"] = doc
        from rag.pipeline import AskResult
        return AskResult(answer="a", sources=[], refused=False, retrieved=[], top_similarity=1.0)

    monkeypatch.setattr(app_module, "ask", fake_ask)
    assert client.post("/ask", json={"question": "q", "doc": "beta.pdf"}).status_code == 200
    assert seen["doc"] == "beta.pdf"
