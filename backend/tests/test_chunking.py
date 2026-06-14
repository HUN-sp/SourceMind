"""Chunking & provenance tests — the brief asks specifically for these.

Covers: page-boundary integrity, provenance carried onto chunks, the token
budget, period-header detection, and statement-metadata derivation.
"""
from rag.chunk import chunk_page, chunk_pages, count_tokens, _detect_period_header
from rag.config import CONFIG
from rag.extract import Page, _detect_statement_meta


def test_chunks_never_cross_a_page_and_keep_provenance():
    p1 = Page(doc="d.pdf", page=1, text="Alpha one.\n\nAlpha two.", source="text-layer", doc_label="X")
    p2 = Page(doc="d.pdf", page=2, text="Beta block.", source="ocr", doc_label="X")
    chunks = chunk_pages([p1, p2])

    assert chunks, "expected at least one chunk"
    # no chunk mixes pages; provenance preserved
    for c in chunks:
        assert c.doc == "d.pdf"
        assert c.page in (1, 2)
        assert c.source in ("text-layer", "ocr")
        # the chunk's page matches the source page's provenance
        assert c.source == ("text-layer" if c.page == 1 else "ocr")
    # chunk_index is sequential within the document, starting at 0
    idxs = [c.chunk_index for c in chunks]
    assert idxs[0] == 0 and idxs == sorted(idxs)


def test_chunks_respect_the_token_budget():
    big = " ".join(f"word{i}" for i in range(2000))  # forces multiple chunks
    chunks = chunk_page(Page(doc="d.pdf", page=1, text=big, source="text-layer"))
    assert len(chunks) > 1
    # body is packed to <= max; allow headroom for an attached header/label
    assert all(count_tokens(c.text) <= CONFIG.chunk_max_tokens + 150 for c in chunks)


def test_period_header_is_detected_for_a_table_page():
    text = (
        "Quarter ended Year ended\n"
        "Particulars 31.03.2026 31.03.2025\n"
        "Unaudited Audited\n"
        "1 Net Profit 100 200"
    )
    header = _detect_period_header(text)
    assert header is not None
    assert "31.03.2026" in header


def test_prose_page_has_no_table_header():
    assert _detect_period_header("This is a plain note with no dated table.") is None


def test_statement_metadata_derived_from_header():
    assert _detect_statement_meta(
        "STANDALONE FINANCIAL RESULTS FOR THE QUARTER AND YEAR ENDED MARCH 31, 2026"
    ) == ("standalone", "March 31, 2026")
    assert _detect_statement_meta(
        "CONSOLIDATED FINANCIAL RESULTS FOR THE QUARTER ENDED JUNE 30, 2025"
    ) == ("consolidated", "June 30, 2025")
    assert _detect_statement_meta("a paragraph with no results header") is None
