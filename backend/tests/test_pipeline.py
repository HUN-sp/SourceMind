"""Grounding-pipeline tests using an INJECTED fake LLM — no Groq tokens spent.

`ask(..., generate=fake)` lets us drive the two grounding gates deterministically:
the model is the only non-local piece, so faking it exercises the whole pipeline
(retrieve -> gate -> parse -> cite) offline.
"""
import json

import pytest

from rag.config import CONFIG
from rag.pipeline import REFUSAL_MESSAGE, ask

pytestmark = pytest.mark.skipif(
    not (CONFIG.index_dir / "embeddings.npy").exists(),
    reason="index not built — run `python ingest.py` first",
)

ANSWERABLE = "What is the gross NPA ratio?"


def _fake_llm(payload: dict):
    return lambda system, user: json.dumps(payload)


def test_answers_and_cites_when_the_model_finds_it():
    r = ask(ANSWERABLE, generate=_fake_llm(
        {"not_covered": False, "answer": "1.15%", "used_passages": [1]}
    ))
    assert not r.refused
    assert "1.15%" in r.answer
    assert r.sources, "an answered question must carry at least one citation"
    assert r.sources[0].page >= 1


def test_refuses_when_the_model_flags_not_covered():
    r = ask(ANSWERABLE, generate=_fake_llm(
        {"not_covered": True, "answer": "", "used_passages": []}
    ))
    assert r.refused
    assert r.answer == REFUSAL_MESSAGE
    assert r.sources == []


def test_fails_safe_on_unparseable_model_output():
    # Garbage from the model must lead to a refusal, never a guess.
    r = ask(ANSWERABLE, generate=lambda system, user: "<<not json at all>>")
    assert r.refused
    assert r.answer == REFUSAL_MESSAGE
