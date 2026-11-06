"""Abuse limits on /ask: bounded question size and a per-client rate limit."""
import json

from fastapi.testclient import TestClient

import app as app_module
from app import app
from rag.ratelimit import SlidingWindowLimiter

client = TestClient(app)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_limiter_allows_up_to_limit_then_blocks_then_recovers():
    clock = Clock()
    lim = SlidingWindowLimiter(limit=3, window_seconds=60, clock=clock)
    assert [lim.check("a")[0] for _ in range(3)] == [True, True, True]
    ok, retry = lim.check("a")
    assert not ok and 0 < retry <= 60
    assert lim.check("b")[0], "other clients are unaffected"
    clock.t += 61
    assert lim.check("a")[0], "window slid past the old hits"


def test_limiter_zero_means_disabled():
    lim = SlidingWindowLimiter(limit=0, window_seconds=60)
    assert all(lim.check("a")[0] for _ in range(1000))


def test_limiter_forgets_idle_clients():
    clock = Clock()
    lim = SlidingWindowLimiter(limit=1, window_seconds=10, clock=clock)
    lim.check("a")
    clock.t += 11
    lim.check("b")
    assert "a" not in lim._hits


def test_overlong_question_is_rejected_before_any_work(monkeypatch):
    called = []
    monkeypatch.setattr(app_module, "ask", lambda *a, **k: called.append(1))
    resp = client.post("/ask", json={"question": "x" * (app_module.CONFIG.max_question_chars + 1)})
    assert resp.status_code == 422
    assert not called


def test_ask_is_rate_limited_with_retry_after(tiny_index, monkeypatch):
    from rag import pipeline

    fake = lambda s, u: json.dumps({"not_covered": True, "answer": "", "used_passages": []})
    monkeypatch.setattr(app_module, "ask", lambda q, doc=None: pipeline.ask(q, doc=doc, generate=fake))
    monkeypatch.setattr(app_module, "limiter", SlidingWindowLimiter(limit=2, window_seconds=60))
    codes = [client.post("/ask", json={"question": "gross NPA"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    resp = client.post("/ask", json={"question": "gross NPA"})
    assert resp.status_code == 429
    assert int(resp.headers["retry-after"]) >= 1
    assert "too many" in resp.json()["detail"].lower()
