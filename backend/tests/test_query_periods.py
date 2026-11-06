"""Questions name a period ("March 31, 2026", "FY2026", "Q2 FY2026") and sometimes a
statement type ("consolidated"). Those should steer WHICH filing/column wins as a
soft bonus on chunk metadata — and be stripped from the text used for semantic /
keyword matching, where they only match the doc-label prefix every chunk shares."""
import dataclasses

import pytest

import rag.store as store
from rag.store import parse_query_periods, query_statement


@pytest.mark.parametrize("q,keys", [
    ("net profit for the full year ended March 31, 2026?", {20260331}),
    ("What was it on 30 September 2025", {20250930}),
    ("as of Sept 30th, 2025", {20250930}),
    ("balance in December 2025", {20251231}),
    ("results for 31.12.2025", {20251231}),
    ("net profit for FY2026", {20260331}),
    ("net profit for FY 26", {20260331}),
    ("in the financial year 2025", {20250331}),
    ("interest income for Q2 FY2026", {20250930}),
    ("Q4 FY26 profit", {20260331}),
    ("Q1 of FY2026", {20250630}),
    ("Q3 FY2026", {20251231}),
    ("How did profit trend across Q2, Q3, and Q4 of FY2026?", {20250930, 20251231, 20260331}),
    ("compare Q1 & Q2 FY26", {20250630, 20250930}),
    ("interest income (July-September 2025)", {20250930}),
    ("compare September 30, 2025 and March 31, 2026", {20250930, 20260331}),
    ("What is the gross NPA ratio?", set()),
    ("it may happen", set()),
    ("founded in 1994", set()),
])
def test_parse_query_periods_keys(q, keys):
    assert parse_query_periods(q)[0] == keys


def test_period_expressions_are_stripped_from_the_search_text():
    keys, rest = parse_query_periods("What is HDFC Bank's Capital Adequacy Ratio as of March 31, 2026?")
    assert keys == {20260331}
    assert "2026" not in rest and "March" not in rest
    assert "Capital Adequacy Ratio" in rest
    _, rest = parse_query_periods("interest income for Q2 FY2026 (July-September 2025)?")
    assert "()" not in rest and "2025" not in rest and "interest income" in rest
    assert parse_query_periods("no period here")[1] == "no period here"


@pytest.mark.parametrize("q,expected", [
    ("consolidated net profit", "consolidated"),
    ("HDFC Bank Group's net profit", "consolidated"),
    ("standalone net profit", "standalone"),
    ("net profit", None),
    ("standalone vs consolidated net profit", None),     # both named: no preference
])
def test_query_statement(q, expected):
    assert query_statement(q) == expected


# --- effect on search (synthetic index: alpha = standalone/Mar-2026, beta = consolidated/Sep-2025,
# both containing a "Gross NPAs to Gross Advances" row) ---------------------------------------
def test_named_period_selects_that_filing(tiny_index):
    q = "Gross NPAs to Gross Advances"
    assert store.search(f"{q} as of March 31, 2026", top_k=1)[0].doc == "alpha.pdf"
    assert store.search(f"{q} as of September 30, 2025", top_k=1)[0].doc == "beta.pdf"


def test_named_statement_type_selects_that_statement(tiny_index):
    q = "Gross NPAs to Gross Advances"
    assert store.search(f"consolidated {q}", top_k=1)[0].statement_type == "consolidated"
    assert store.search(f"standalone {q}", top_k=1)[0].statement_type == "standalone"


def test_boost_can_be_disabled(tiny_index, monkeypatch):
    monkeypatch.setattr(store, "CONFIG", dataclasses.replace(store.CONFIG, period_boost=0.0))
    # no crash, still returns results
    assert store.search("Gross NPAs to Gross Advances as of March 31, 2026", top_k=2)


# --- focusing the query on terms that actually discriminate ----------------------
@pytest.fixture
def big_index(tmp_path, monkeypatch):
    """60 synthetic pages: every one says 'quarter ended results', each has its own
    metric word; one mentions Dubai."""
    import rag.store as st
    from rag.chunk import chunk_pages
    from rag.extract import Page

    cfg = dataclasses.replace(st.CONFIG, index_dir=tmp_path / "index")
    monkeypatch.setattr(st, "CONFIG", cfg)
    st.reload_index()
    pages = [
        Page(doc="d.pdf", page=i + 1, source="text-layer", doc_label="Standalone Results — March 31, 2026",
             text=f"quarter ended results metric{i} value {i}.5 text for the period")
        for i in range(60)
    ]
    pages[7] = dataclasses.replace(pages[7], text="quarter ended results the Dubai branch notice")
    st.index_chunks(chunk_pages(pages))
    yield
    st.reload_index()


def test_focus_query_drops_corpus_ubiquitous_terms(big_index):
    kept = store.focus_query("What did the Dubai branch results show for the quarter ended?").split()
    assert "dubai" in kept
    assert not {"results", "quarter", "ended", "what", "the", "for"} & set(kept)


def test_focus_query_never_returns_empty(big_index):
    assert store.focus_query("results quarter ended").strip() != ""   # all terms common -> original text
    assert store.focus_query("???").strip() != ""


def test_focus_applies_to_search_so_the_metric_word_decides(big_index):
    top = store.search("results for the quarter ended metric13", top_k=1)[0]
    assert top.page == 14


def test_focus_is_skipped_on_a_tiny_corpus(tiny_index):
    q = "What is the gross NPA ratio?"
    # ratio-based filtering would remove every term here; the guard must leave the query intact
    assert store._common_terms(store.generation(), 0.25) == frozenset()


def test_focus_can_be_disabled(big_index, monkeypatch):
    monkeypatch.setattr(store, "CONFIG", dataclasses.replace(store.CONFIG, focus_query=False))
    q = "What did the Dubai branch results show?"
    assert store.focus_query(q) == q


# --- document frequency is measured over page-level chunks, not row chunks ---------
def test_common_terms_ignore_row_chunks(monkeypatch):
    import numpy as np

    pages = [{"text": f"hdfc bank page {i}", "chunk_type": "table"} for i in range(60)]
    rows = [{"text": f"row label {i}", "chunk_type": "row"} for i in range(600)]   # no 'bank' anywhere
    monkeypatch.setattr(store, "_load", lambda: (np.zeros((660, 2)), pages + rows))
    store._common_terms.cache_clear()
    common = store._common_terms(0, 0.5)
    store._common_terms.cache_clear()
    assert {"hdfc", "bank", "page"} <= common      # in 100% of page-level chunks (only 9% of all chunks)
    assert "label" not in common


# --- rows and page chunks compete in separate rankings -----------------------------
def test_results_mix_row_and_page_chunks(tiny_index):
    kinds = {r.chunk_type for r in store.search("Capital Adequacy Ratio", top_k=4)}
    assert "row" in kinds and kinds & {"table", "prose"}


# --- weights in rank fusion ---------------------------------------------------------
def test_rrf_weights_scale_a_lists_influence():
    import numpy as np

    cand = np.arange(2)
    lists = [np.array([0, 1]), np.array([1, 0])]
    assert store._rrf_fuse(lists, cand, top_k=2, weights=[1.0, 0.2])[0] == 0
    assert store._rrf_fuse(lists, cand, top_k=2, weights=[0.2, 1.0])[0] == 1
    assert store._rrf_fuse(lists, cand, top_k=2)[0] in (0, 1)          # default: equal weights, no error


def test_row_weight_default_is_a_sane_positive_number():
    assert 0 < store.CONFIG.row_weight <= 1
