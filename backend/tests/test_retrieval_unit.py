"""Hermetic tests of retrieval logic (synthetic index, fake embedder): ranking
helpers, hybrid fusion, recency tiebreak, document filter, and the pipeline's
pre-LLM grounding gate."""
import json

import numpy as np
import pytest

import rag.store as store
from rag.pipeline import REFUSAL_MESSAGE, ask


# --- pure helpers ----------------------------------------------------------
def test_period_key_orders_dates():
    k = store._period_key
    assert k("March 31, 2026") == 20260331
    assert k("June 30, 2025") < k("September 30, 2025") < k("March 31, 2026")
    assert k("") == 0 and k("no date here") == 0


@pytest.mark.parametrize("q,expected", [
    ("What was net profit for FY2026?", False),   # FY2026 has no word boundary before 2026 -> not a bare year
    ("What was net profit in 2025?", True),
    ("What was it in March?", True),
    ("What is the gross NPA ratio?", False),
])
def test_query_names_period(q, expected):
    assert store._query_names_period(q) is expected


def test_query_tokens_drop_stopwords_and_bare_numbers_but_never_return_empty():
    assert store._query_tokens("What is the Capital Adequacy Ratio as of 2026?") == ["capital", "adequacy", "ratio"]
    assert store._query_tokens("the of 2026") != []  # falls back to all tokens


def test_rrf_fuse_prefers_chunks_ranked_well_in_both_lists():
    cand = np.arange(4)
    fused = store._rrf_fuse([np.array([0, 1, 2, 3]), np.array([1, 0, 3, 2])], cand, top_k=4)
    assert set(fused[:2]) == {0, 1} and set(fused[2:]) == {2, 3}


def test_rrf_fuse_bonus_breaks_ties():
    cand = np.arange(2)
    fused = store._rrf_fuse([np.array([0, 1]), np.array([1, 0])], cand, top_k=2, bonus={1: 0.01})
    assert fused[0] == 1


# --- search over the synthetic index --------------------------------------
def test_search_finds_the_row_with_the_literal_label(tiny_index):
    res = store.search("Capital Adequacy Ratio", top_k=3)
    assert "Capital Adequacy Ratio" in res[0].text
    assert res[0].doc == "alpha.pdf"


def test_search_similarity_is_true_cosine_and_bounded(tiny_index):
    res = store.search("Dubai branch regulator", top_k=4)
    assert all(-1e-6 <= r.similarity <= 1.0001 for r in res)
    assert res[0].page == 2  # the Dubai note


def test_document_filter_and_unknown_document(tiny_index):
    only_beta = store.search("gross npa", top_k=8, doc="beta.pdf")
    assert only_beta and {r.doc for r in only_beta} == {"beta.pdf"}
    assert store.search("gross npa", doc="nope.pdf") == []


def test_top_k_is_respected_and_results_carry_metadata(tiny_index):
    res = store.search("gross npa", top_k=2)
    assert len(res) == 2
    assert all(r.doc_label and r.statement_type and r.fiscal_period for r in res)


def test_recency_tiebreak_prefers_latest_filing_only_when_no_period_named(tiny_index, monkeypatch):
    import dataclasses

    monkeypatch.setattr(store, "CONFIG", dataclasses.replace(store.CONFIG, recency_boost=1.0))
    latest = store.search("Gross NPAs to Gross Advances", top_k=1, hybrid=True)[0]
    assert latest.fiscal_period == "March 31, 2026"
    named = store.search("Gross NPAs to Gross Advances September 2025", top_k=1, hybrid=True)[0]
    assert named.fiscal_period == "September 30, 2025"  # explicit period disables the bonus


# --- pipeline gate #1 ------------------------------------------------------
def test_gate_one_refuses_without_calling_the_llm(tiny_index):
    def must_not_run(system, user):
        raise AssertionError("LLM must not be called when nothing relevant was retrieved")

    r = ask("zzzz qqqq xxxx", generate=must_not_run)
    assert r.refused and r.answer == REFUSAL_MESSAGE and r.top_similarity < 0.30


def test_unknown_document_filter_refuses(tiny_index):
    r = ask("gross npa", doc="nope.pdf", generate=lambda s, u: "{}")
    assert r.refused


def test_prompt_orders_passages_by_document_position(tiny_index):
    seen = {}

    def spy(system, user):
        seen["user"] = user
        return json.dumps({"not_covered": False, "answer": "ok", "used_passages": [1]})

    ask("Gross NPAs to Gross Advances ratio", generate=spy, top_k=8)
    docs = [line.split(",")[0].split("] ")[1] for line in seen["user"].splitlines() if line.startswith("[")]
    assert docs == sorted(docs)
