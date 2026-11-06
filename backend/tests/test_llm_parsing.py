"""Fail-safe handling of whatever the LLM returns. The contract (pipeline docs):
unparseable / malformed output -> refuse (or cite the best hit), NEVER crash."""
import json

import pytest

from rag.pipeline import REFUSAL_MESSAGE, ask, parse_llm_json

Q = "What is the gross NPA ratio to gross advances?"


def _gen(raw):
    return lambda system, user: raw if isinstance(raw, str) else json.dumps(raw)


def test_parse_accepts_json_wrapped_in_stray_text():
    assert parse_llm_json('Sure! {"answer": "x"} hope that helps') == {"answer": "x"}


@pytest.mark.parametrize("raw", ["[1, 2]", '"just a string"', "42", "null", "true"])
def test_parse_rejects_non_object_json(raw):
    with pytest.raises(ValueError):
        parse_llm_json(raw)


@pytest.mark.parametrize("raw", ["[1, 2]", '"text"', "null", "", "{not json"])
def test_ask_refuses_instead_of_crashing_on_non_object_output(tiny_index, raw):
    r = ask(Q, generate=_gen(raw))
    assert r.refused and r.answer == REFUSAL_MESSAGE and r.sources == []


@pytest.mark.parametrize("used", [3, "1, 2", None, {"a": 1}, [None, "x", 99, -4, 1.5], True])
def test_ask_tolerates_malformed_used_passages(tiny_index, used):
    r = ask(Q, generate=_gen({"not_covered": False, "answer": "1.15%", "used_passages": used}))
    assert not r.refused
    assert r.answer == "1.15%"
    assert r.sources, "falls back to citing the best-matching chunk"


def test_ask_cites_the_numbered_passages_the_model_used(tiny_index):
    r = ask(Q, generate=_gen({"not_covered": False, "answer": "a", "used_passages": [1, "2"]}))
    assert len(r.sources) == 2


@pytest.mark.parametrize("flag,refused", [(True, True), ("true", True), ("false", False), (False, False), (None, False)])
def test_not_covered_flag_is_interpreted_strictly(tiny_index, flag, refused):
    r = ask(Q, generate=_gen({"not_covered": flag, "answer": "1.15%", "used_passages": [1]}))
    assert r.refused is refused
