"""Row chunks render each value next to its column ("Year ended March 31, 2026: 74671.29")
so the LLM never has to infer which of five bare numbers is which. Column dates come from
the filing's detected period (reliable) — NOT from the OCR'd header dates (e.g. real OCR
read the year-ago column as 31.03.2026)."""
import pytest

from rag.chunk import chunk_page, column_labels, embedding_text
from rag.extract import Page

Q4_HEADER = "Quarterended Yearended Particulars 31.03.2026 31.12.2026 31.03.2025 31.03.2026 31.03.2026"  # real OCR, dates garbled


def test_q4_filing_quarter_and_year_columns():
    assert column_labels(Q4_HEADER, 5, "March 31, 2026") == [
        "Quarter ended March 31, 2026", "Quarter ended December 31, 2025", "Quarter ended March 31, 2025",
        "Year ended March 31, 2026", "Year ended March 31, 2025",
    ]


def test_q2_filing_has_a_half_year_group():
    assert column_labels("Quarter ended Half year ended Particulars 30.09.2025", 5, "September 30, 2025") == [
        "Quarter ended September 30, 2025", "Quarter ended June 30, 2025", "Quarter ended September 30, 2024",
        "Half year ended September 30, 2025", "Half year ended September 30, 2024",
    ]


def test_q3_filing_has_a_nine_months_group():
    labels = column_labels("Quarter ended Nine months ended 31.12.2025", 5, "December 31, 2025")
    assert labels[3:] == ["Nine months ended December 31, 2025", "Nine months ended December 31, 2024"]
    assert labels[1] == "Quarter ended September 30, 2025"


def test_q1_filing_single_quarter_group():
    assert column_labels("Quarter ended 30.06.2025", 3, "June 30, 2025") == [
        "Quarter ended June 30, 2025", "Quarter ended March 31, 2025", "Quarter ended June 30, 2024",
    ]


def test_three_groups():
    labels = column_labels("Quarter ended Half year ended Year ended", 6, "September 30, 2025")
    assert [l.split(" ended")[0] for l in labels] == ["Quarter"] * 3 + ["Half year"] * 2 + ["Year"]


@pytest.mark.parametrize("header,n,period", [
    (Q4_HEADER, 4, "March 31, 2026"),            # value count doesn't fit the layout -> don't guess
    (Q4_HEADER, 5, ""),                          # no detected period
    ("Particulars 31.03.2026", 5, "March 31, 2026"),   # no group words
    (None, 5, "March 31, 2026"),
])
def test_returns_none_instead_of_guessing(header, n, period):
    assert column_labels(header, n, period) is None


# --- in row chunks ----------------------------------------------------------------
PAGE = (
    "Quarterended Yearended Particulars 31.03.2026 31.12.2026 31.03.2025 31.03.2026 31.03.2026\n"
    "12 Net Profit from ordinary activities after tax 19221.05 18663.75 17616.14 74671.29 67347.36 "
    "14 Net Profit for the period 19221.05 18663.75 17616.14 74671.29 67347.36 "
    "(ii)Capital Adequacy Ratio 19.71% 19.87% 19.55% 19.71% 19.55%"
)


def _rows(period="March 31, 2026", text=PAGE):
    page = Page(doc="d.pdf", page=1, text=text, source="ocr", statement_type="standalone",
                fiscal_period=period, doc_label="Standalone Results — March 31, 2026")
    return [c for c in chunk_page(page) if c.chunk_type == "row"]


def test_row_chunk_pairs_every_value_with_its_column():
    row = next(c for c in _rows() if "Capital Adequacy" in c.text)
    last = row.text.splitlines()[-1]
    assert "Year ended March 31, 2026: 19.71%" in last
    assert "Quarter ended December 31, 2025: 19.87%" in last
    assert "Year ended March 31, 2025: 19.55%" in last


def test_net_profit_year_column_is_unambiguous():
    row = next(c for c in _rows() if "Net Profit for the period" in c.text)
    assert "Year ended March 31, 2026: 74671.29" in row.text
    assert "Quarter ended March 31, 2026: 19221.05" in row.text


def test_rendered_rows_drop_the_ambiguous_ocr_header():
    row = next(c for c in _rows() if "Capital Adequacy" in c.text)
    assert "Quarterended" not in row.text and "31.12.2026" not in row.text
    assert row.text.startswith("Standalone Results — March 31, 2026\n")


def test_embedding_text_is_still_just_the_label():
    row = next(c for c in _rows() if "Capital Adequacy" in c.text)
    emb = embedding_text(row)
    assert "Capital Adequacy Ratio" in emb
    assert "19.71" not in emb and "Year ended" not in emb and "Quarter ended" not in emb


def test_unrenderable_rows_keep_the_raw_header_and_values():
    rows = _rows(period="")                           # no period -> cannot label columns
    row = next(c for c in rows if "Capital Adequacy" in c.text)
    assert "Quarterended" in row.text and "19.71%" in row.text
