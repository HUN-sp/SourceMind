"""`.env.example` must not silently change behaviour: a user who copies it verbatim
should get the same settings as the code defaults (DECISIONS §6: reranker is OFF
by default because it hurts numeric value rows)."""
import importlib
import re
from pathlib import Path

import pytest

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"
TUNABLES = {
    "TOP_K": "top_k",
    "MIN_SIMILARITY": "min_similarity",
    "USE_HYBRID": "use_hybrid",
    "RRF_K": "rrf_k",
    "RECENCY_BOOST": "recency_boost",
    "USE_RERANKER": "use_reranker",
    "RERANKER_MODEL": "reranker_model",
    "RERANK_CANDIDATES": "rerank_candidates",
    "LLM_MODEL": "llm_model",
    "EMBEDDING_MODEL": "embedding_model",
    "MAX_QUESTION_CHARS": "max_question_chars",
    "RATE_LIMIT_PER_MINUTE": "rate_limit_per_minute",
    "CACHE_SIZE": "cache_size",
    "ROW_CHUNKS": "row_chunks",
    "ROW_WEIGHT": "row_weight",
    "PERIOD_BOOST": "period_boost",
    "FOCUS_QUERY": "focus_query",
    "COMMON_TERM_RATIO": "common_term_ratio",
}


def _parse_env_example() -> dict[str, str]:
    out = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


@pytest.fixture
def default_config(monkeypatch):
    """Config as built with NO environment overrides and no developer .env."""
    import dotenv
    import rag.config as cfg_mod

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for k in TUNABLES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yield importlib.reload(cfg_mod).Config()
    monkeypatch.undo()
    importlib.reload(cfg_mod)


def test_env_example_matches_code_defaults(default_config):
    env = _parse_env_example()
    for key, attr in TUNABLES.items():
        assert key in env, f"{key} missing from .env.example"
        default = getattr(default_config, attr)
        raw = env[key]
        if isinstance(default, bool):
            assert (raw.lower() in ("1", "true", "yes")) == default, key
        elif isinstance(default, (int, float)):
            assert float(raw) == float(default), key
        else:
            assert raw == default, key


def test_env_example_has_no_real_key():
    env = _parse_env_example()
    assert env["GROQ_API_KEY"].startswith("gsk_your_")
