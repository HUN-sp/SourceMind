"""Public-demo features: preset responses, retrieval scores, and the live-LLM quota."""
import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app
from rag import demo
from rag.quota import LLMQuota, QuotaExceeded
from rag.ratelimit import SlidingWindowLimiter

client = TestClient(app)

ANSWER = json.dumps({"not_covered": False, "answer": "It was 1.15%.", "used_passages": [1]})


@pytest.fixture
def live(tiny_index, monkeypatch):
    """Real pipeline over the synthetic index with a counting fake LLM."""
    calls = []

    def fake_llm(system, user):
        calls.append(user)
        return ANSWER

    monkeypatch.setattr(app_module, "llm_generate", fake_llm)
    monkeypatch.setattr(app_module, "limiter", SlidingWindowLimiter(limit=0))
    monkeypatch.setattr(app_module, "answer_cache", app_module.AnswerCache(0))
    monkeypatch.setattr(app_module, "llm_quota", LLMQuota("100/hour", ""))
    monkeypatch.setattr(app_module, "_preset_cache", {})
    return calls


def test_response_exposes_retrieval_scores(live):
    body = client.post("/ask", json={"question": "Gross NPAs to Gross Advances"}).json()
    assert not body["refused"] and live, "the fake LLM answered"
    assert body["retrieval"], "retrieved chunks are reported"
    top = body["retrieval"][0]
    for field in ("cosine", "bm25", "rrf", "vec_rank", "bm25_rank"):
        assert top[field] is not None, field
    rrfs = [c["rrf"] for c in body["retrieval"]]
    assert rrfs == sorted(rrfs, reverse=True), "chunks are listed in final (RRF) order"
    assert any(c["cited"] for c in body["retrieval"]), "the cited chunk is flagged"
    assert body["latency_ms"] is not None and body["top_similarity"] > 0


def test_preset_is_served_from_precomputed_file_without_llm(live, monkeypatch):
    p = demo.PRESETS[0]
    stored = {"answer": "From file.", "sources": [], "refused": False, "retrieval": []}
    monkeypatch.setattr(app_module, "_preset_cache", {p["id"]: stored})
    # Whitespace/case differences still match the preset.
    body = client.post("/ask", json={"question": "  " + p["question"].upper()}).json()
    assert body["answer"] == "From file." and body["cached"] and body["preset"] == p["id"]
    assert not live, "no LLM call for a preset"


def test_preset_without_cache_falls_back_to_live_and_is_tagged(live):
    p = demo.PRESETS[0]
    body = client.post("/ask", json={"question": p["question"]}).json()
    assert body["preset"] == p["id"] and not body["cached"]


def test_quota_counts_only_live_llm_calls(live, monkeypatch):
    monkeypatch.setattr(app_module, "llm_quota", LLMQuota("2/hour", ""))
    stored = {"answer": "From file.", "sources": [], "refused": False, "retrieval": []}
    monkeypatch.setattr(app_module, "_preset_cache", {demo.PRESETS[0]["id"]: stored})
    q = {"question": "Gross NPAs to Gross Advances"}
    assert [client.post("/ask", json=q).status_code for _ in range(2)] == [200, 200]
    resp = client.post("/ask", json=q)
    assert resp.status_code == 429 and int(resp.headers["retry-after"]) >= 1
    assert "live questions" in resp.json()["detail"]
    # Presets keep working after the visitor's live quota is spent.
    assert client.post("/ask", json={"question": demo.PRESETS[0]["question"]}).status_code == 200
    assert len(live) == 2


def test_quota_is_per_ip_and_global():
    q = LLMQuota("1/hour", "2/day")
    q.charge("a")
    with pytest.raises(QuotaExceeded):
        q.charge("a")
    q.charge("b")  # another visitor has their own bucket
    with pytest.raises(QuotaExceeded, match="shared daily"):
        q.charge("c")  # but the global bucket is now full


def test_demo_endpoint_lists_presets():
    body = client.get("/demo").json()
    kinds = [p["id"] for p in body["presets"]]
    assert kinds == ["numeric", "multihop", "trap"]
    assert all(p["question"] and "instant" in p for p in body["presets"])


def test_stale_cache_is_ignored(tmp_path, monkeypatch):
    f = tmp_path / "demo_cache.json"
    f.write_text(json.dumps({"index_sha256": "not-this-index", "responses": {"numeric": {}}}))
    monkeypatch.setattr(demo, "CACHE_FILE", f)
    assert demo.load_cache() == {}
