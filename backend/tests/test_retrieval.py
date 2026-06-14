"""Retrieval tests — run against the real index (no LLM, no Groq tokens).

Skipped automatically if the index hasn't been built yet.
"""
import pytest

from rag.config import CONFIG
from rag.store import search

pytestmark = pytest.mark.skipif(
    not (CONFIG.index_dir / "embeddings.npy").exists(),
    reason="index not built — run `python ingest.py` first",
)


def test_search_returns_at_most_top_k_with_valid_scores():
    res = search("gross NPA ratio", top_k=5)
    assert 0 < len(res) <= 5
    assert all(0.0 <= r.similarity <= 1.0001 for r in res)
    # results are surfaced most-relevant first by cosine when not reranked
    sims = [r.similarity for r in res]
    assert max(sims) >= 0.4  # a real question should match something


def test_known_question_retrieves_the_latest_filing():
    res = search("What is the gross NPA ratio?", top_k=8)
    assert "data_4.pdf" in {r.doc for r in res}  # FY2026 filing should appear


def test_document_filter_restricts_results():
    res = search("net profit", top_k=8, doc="data_2.pdf")
    assert res and all(r.doc == "data_2.pdf" for r in res)


def test_metadata_is_populated_on_results():
    res = search("capital adequacy ratio", top_k=5)
    assert res
    r = res[0]
    assert r.doc_label                      # human-readable citation label present
    assert r.statement_type in ("standalone", "consolidated", "")
    assert r.chunk_type in ("table", "prose", "")
