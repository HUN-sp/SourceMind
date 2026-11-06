"""LLM failures -> clear, safe HTTP responses. Raw provider text (which can carry
org ids, request ids, quota figures) must not be forwarded to API clients."""
import httpx
import groq
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app
from rag import generate as gen
from rag.generate import (
    LLMAuthError, LLMConfigError, LLMError, LLMRateLimitError, LLMUnavailableError,
)

client = TestClient(app)
SECRET_DETAIL = "org_9f3 request_id=req_abc quota=100000 TPD"


def _resp(status, headers=None):
    return httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://x"))


def _client_raising(exc):
    class C:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise exc
    return C()


@pytest.mark.parametrize("exc,expected", [
    (groq.RateLimitError(SECRET_DETAIL, response=_resp(429, {"retry-after": "42"}), body=None), LLMRateLimitError),
    (groq.AuthenticationError(SECRET_DETAIL, response=_resp(401), body=None), LLMAuthError),
    (groq.InternalServerError(SECRET_DETAIL, response=_resp(500), body=None), LLMUnavailableError),
    (groq.APIConnectionError(request=httpx.Request("POST", "https://x")), LLMUnavailableError),
    (groq.APITimeoutError(request=httpx.Request("POST", "https://x")), LLMUnavailableError),
    (ValueError("weird " + SECRET_DETAIL), LLMError),
])
def test_provider_errors_are_translated_and_scrubbed(monkeypatch, exc, expected):
    monkeypatch.setattr(gen, "_client", lambda: _client_raising(exc))
    with pytest.raises(expected) as ei:
        gen.generate("s", "u")
    assert type(ei.value) is expected
    assert "org_9f3" not in str(ei.value) and "req_abc" not in str(ei.value)


def test_rate_limit_error_carries_retry_after(monkeypatch):
    exc = groq.RateLimitError("x", response=_resp(429, {"retry-after": "42"}), body=None)
    monkeypatch.setattr(gen, "_client", lambda: _client_raising(exc))
    with pytest.raises(LLMRateLimitError) as ei:
        gen.generate("s", "u")
    assert ei.value.retry_after == 42


def test_api_key_is_never_echoed(monkeypatch):
    cfg = type("C", (), {"groq_api_key": "gsk_FAKE_TEST_KEY_123", "llm_model": "m"})()
    monkeypatch.setattr(gen, "CONFIG", cfg)
    monkeypatch.setattr(gen, "_client", lambda: _client_raising(ValueError("boom gsk_FAKE_TEST_KEY_123 boom")))
    with pytest.raises(LLMError) as ei:
        gen.generate("s", "u")
    assert "gsk_FAKE_TEST_KEY_123" not in str(ei.value)


@pytest.mark.parametrize("exc,status", [
    (LLMRateLimitError(retry_after=42), 429),
    (LLMRateLimitError(), 429),
    (LLMConfigError("GROQ_API_KEY is not set."), 503),
    (LLMAuthError(), 502),
    (LLMUnavailableError(), 502),
    (LLMError("LLM call failed."), 502),
])
def test_endpoint_maps_llm_errors_to_status_codes(monkeypatch, tiny_index, exc, status):
    def boom(q, doc=None):
        raise exc

    monkeypatch.setattr(app_module, "ask", boom)
    monkeypatch.setattr(app_module, "limiter", app_module.SlidingWindowLimiter(limit=0))
    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == status
    assert resp.json()["detail"]
    if isinstance(exc, LLMRateLimitError) and exc.retry_after:
        assert resp.headers["retry-after"] == "42"
