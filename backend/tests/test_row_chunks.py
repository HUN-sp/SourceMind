"""Row-level chunks: each table row (label + values) becomes its own small chunk so
terse numeric rows aren't buried in a dense page-sized blob (the retrieval weakness
documented in DECISIONS.md §3). The text below mimics real OCR output: glued words,
rows run together on one line, a row cut across a blank-line block boundary."""
import dataclasses

import rag.chunk as chunk
from rag.chunk import chunk_page, split_rows
from rag.extract import Page

OCR = (
    "Quarterended Yearended Particulars 31.03.2026 31.12.2026 31.03.2025 31.03.2026 31.03.2026\n"
    "11 Tax Expense(Refernote 18) 5972.30 5606.19 5727.51 20497.40 21130.70 12 Net Profit from ordlnary "
    "actlvities after tax{10)-(11) 19221.05 18663.75 17616.14 74671.29 67347.36 13 Extraordinaryitems(netof "
    "taxexpense) 14 Net Profit forthe period (12)-(13) 19221.05 18663.75 17616.14 74671.29 67347.36\n\n"
    "15 Paid up equity share capital (Face Value of 1/- each) 1539.34 1538.46 765.22 1539.34 765.22 "
    "(i)Percentage of shares held by Government of India Nil Nit IIN IIN Nil "
    "(ll)Capital Adequacy Ratio 19.71% 19.87% 19.55% 19.71% 19.55%\n"
    "(c)% of Gross NPAs to Gross Advances 1.15% 1.24% 1.33% 1.15% 1.33%"
)


def _page(text=OCR):
    return Page(doc="d.pdf", page=1, text=text, source="ocr", statement_type="standalone",
                fiscal_period="March 31, 2026", doc_label="Standalone Results — March 31, 2026")


def test_split_rows_isolates_each_row_with_its_own_values():
    rows = split_rows(OCR)
    by_label = {k: r for k in ("Net Profit forthe period", "Capital Adequacy", "Gross NPAs", "Tax Expense")
                for r in rows if k in r}
    assert "74671.29" in by_label["Net Profit forthe period"]
    assert "19.71%" not in by_label["Net Profit forthe period"]          # CAR is a different row
    assert "19.71%" in by_label["Capital Adequacy"] and "74671.29" not in by_label["Capital Adequacy"]
    assert "1.15%" in by_label["Gross NPAs"]
    assert "5972.30" in by_label["Tax Expense"]


def test_header_date_row_is_not_emitted_as_a_data_row():
    assert not any(r.startswith("Quarterended") or "Particulars 31.03.2026" in r for r in split_rows(OCR))


def test_every_value_survives_splitting():
    import re
    no_dates = re.sub(r"\b\d{1,2}\.\d{2}\.\d{4}\b", "", OCR)  # header dates are intentionally dropped
    vals = set(re.findall(r"\d+\.\d+%?", no_dates))
    text = " ".join(split_rows(OCR))
    assert all(v in text for v in vals), [v for v in vals if v not in text]


def test_a_row_cut_by_a_blank_line_stays_whole():
    text = "Net Profit for the period 100.5 200.5\n\n300.5 400.5 500.5\nTotal Income 1.5 2.5 3.5"
    rows = split_rows(text)
    assert any("Net Profit" in r and "500.5" in r for r in rows), rows


def test_row_chunks_carry_label_columns_and_provenance():
    out = [c for c in chunk_page(_page()) if c.chunk_type == "row"]
    assert out
    np_row = next(c for c in out if "Net Profit forthe period" in c.text)
    assert np_row.text.startswith("Standalone Results — March 31, 2026")
    assert "Year ended March 31, 2026: 74671.29" in np_row.text   # values paired with their columns (see test_column_labels.py)
    assert np_row.doc == "d.pdf" and np_row.page == 1 and np_row.source == "ocr"
    assert np_row.statement_type == "standalone" and np_row.fiscal_period == "March 31, 2026"
    assert len({c.id for c in out}) == len(out)     # ids unique


def test_page_level_chunks_are_still_emitted():
    kinds = {c.chunk_type for c in chunk_page(_page())}
    assert "table" in kinds and "row" in kinds


def test_prose_pages_get_no_row_chunks():
    prose = "The Dubai branch received a decision notice in 2026 from the regulator, effective immediately."
    assert [c for c in chunk_page(_page(prose)) if c.chunk_type == "row"] == []


def test_row_chunks_can_be_disabled(monkeypatch):
    monkeypatch.setattr(chunk, "CONFIG", dataclasses.replace(chunk.CONFIG, row_chunks=False))
    assert all(c.chunk_type != "row" for c in chunk_page(_page()))


def test_overlong_label_is_trimmed_not_dropped():
    text = "word " * 200 + "Capital Adequacy Ratio 19.71% 19.87% 19.55%"
    row = split_rows(text)[0]
    assert "Capital Adequacy Ratio" in row and "19.55%" in row and len(row.split()) < 120


# --- embedding text for rows ------------------------------------------------
def test_row_embedding_text_is_the_label_only_without_prefix_or_numbers():
    from rag.chunk import embedding_text

    row = next(c for c in chunk_page(_page()) if c.chunk_type == "row" and "Net Profit forthe period" in c.text)
    emb = embedding_text(row)
    assert "Net Profit forthe period" in emb
    assert "Standalone Results" not in emb and "Quarterended" not in emb   # shared prefix dropped
    assert "74671.29" not in emb and "19221.05" not in emb                  # numbers dropped
    # page-level chunks are embedded exactly as stored
    page_chunk = next(c for c in chunk_page(_page()) if c.chunk_type == "table")
    assert embedding_text(page_chunk) == page_chunk.text


def test_placeholder_only_row_is_split_from_the_next_label():
    rows = split_rows(OCR)
    car = next(r for r in rows if "Capital Adequacy" in r)
    assert "Percentage of shares" not in car          # previously glued onto the CAR row
    assert any("Percentage of shares held by Government of India Nil" in r for r in rows)
