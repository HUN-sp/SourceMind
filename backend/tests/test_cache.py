"""E2: in-memory answer cache — saves LLM tokens on repeated questions."""
from fastapi.testclient import TestClient

import app as app_module
import rag.store as store
from app import app
from rag.cache import AnswerCache
from rag.pipeline import AskResult, Source

client = TestClient(app)


# --- unit ------------------------------------------------------------------
def test_key_normalises_case_and_whitespace_but_not_doc_or_generation():
    k = AnswerCache.key
    assert k("What  is NPA?\n", None, 0) == k("what is npa?", None, 0)
    assert k("q", "a.pdf", 0) != k("q", "b.pdf", 0)
    assert k("q", None, 0) != k("q", None, 1)


def test_lru_eviction_and_recency():
    c = AnswerCache(maxsize=2)
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") == 1          # touch a -> b is now least recent
    c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3


def test_zero_size_disables_caching():
    c = AnswerCache(maxsize=0)
    c.put("a", 1)
    assert c.get("a") is None and len(c) == 0


# --- endpoint --------------------------------------------------------------
def _setup(monkeypatch, refused=False):
    calls = []

    def fake_ask(q, doc=None):
        calls.append((q, doc))
        src = [] if refused else [Source(doc="d.pdf", page=1, snippet="s", similarity=0.9, fiscal_period="March 31, 2026")]
        return AskResult(answer="nope" if refused else "1.15%", sources=src, refused=refused, retrieved=[], top_similarity=0.9)

    monkeypatch.setattr(app_module, "ask", fake_ask)
    monkeypatch.setattr(app_module, "limiter", app_module.SlidingWindowLimiter(limit=0))
    monkeypatch.setattr(app_module, "answer_cache", AnswerCache(maxsize=8))
    return calls


def test_repeated_question_is_served_from_cache(tiny_index, monkeypatch):
    calls = _setup(monkeypatch)
    first = client.post("/ask", json={"question": "What is NPA?"}).json()
    second = client.post("/ask", json={"question": "  what is   npa? "}).json()
    assert len(calls) == 1
    assert first["cached"] is False and second["cached"] is True
    assert second["answer"] == first["answer"] and second["sources"] == first["sources"]


def test_refusals_are_not_cached(tiny_index, monkeypatch):
    calls = _setup(monkeypatch, refused=True)
    client.post("/ask", json={"question": "q"}); client.post("/ask", json={"question": "q"})
    assert len(calls) == 2   # a transient LLM hiccup must not become a sticky refusal


def test_different_documents_do_not_share_cache_entries(tiny_index, monkeypatch):
    calls = _setup(monkeypatch)
    client.post("/ask", json={"question": "q"})
    client.post("/ask", json={"question": "q", "doc": "alpha.pdf"})
    client.post("/ask", json={"question": "q", "doc": "alpha.pdf"})
    assert [c[1] for c in calls] == [None, "alpha.pdf"]


def test_reindexing_invalidates_the_cache(tiny_index, monkeypatch):
    calls = _setup(monkeypatch)
    client.post("/ask", json={"question": "q"})
    store.reload_index()
    client.post("/ask", json={"question": "q"})
    assert len(calls) == 2


def test_health_reports_cache_stats(tiny_index, monkeypatch):
    _setup(monkeypatch)
    client.post("/ask", json={"question": "q"}); client.post("/ask", json={"question": "q"})
    body = client.get("/health").json()
    assert body["cache"] == {"entries": 1, "hits": 1, "misses": 1, "max_entries": 8}
