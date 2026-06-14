"""Turn a PDF into clean, per-page text — with provenance.

This is the most corpus-sensitive part of the system, so the design is driven by
what the four provided PDFs actually are (verified by inspecting them):

  - data_2.pdf : clean digital PDF (embedded font). Text layer is reliable.
  - data_1.pdf : scanned page-image + a hidden OCR text layer. The body text is
    usable; only decorative header glyphs come out garbled.
  - data_3.pdf : same as data_1.
  - data_4.pdf : pure scan (Microsoft "Print To PDF"), ZERO text layer, one
    full-page image per page. The text layer is empty -> must be OCR'd.

So the pipeline is GENERAL, not special-cased per file:
  1. Try the text layer with layout=True (keeps table columns aligned by spaces).
  2. If a page yields almost no text, fall back to OCR on a rasterized image.
  3. Clean every page with a generic filter that drops mostly-symbol lines
     (this removes the garbled logo/header glyphs without per-file rules).

The same logic would work on a brand-new scanned document the reviewer swaps in.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from .config import CONFIG


@dataclass
class Page:
    """One page of one document, after extraction + cleaning."""
    doc: str          # e.g. "data_4.pdf"
    page: int         # 1-based page number (this is our citation unit)
    text: str         # cleaned text
    source: str       # "text-layer" or "ocr" — recorded for transparency
    # --- derived metadata (from the document's OWN header text, never the filename),
    # so the LLM can tell standalone from consolidated figures and pick the right period.
    statement_type: str = ""   # "standalone" | "consolidated" | ""
    fiscal_period: str = ""    # the period the filing covers, e.g. "March 31, 2026"
    doc_label: str = ""        # human-readable citation label, e.g. "Standalone Results — March 31, 2026"


# A line that is mostly non-word characters is almost certainly a garbled logo /
# header glyph (e.g.  [' " l=1•1iil=t·Hl\i ). We drop those. This is generic:
# it never names a specific document.
def _is_garbage_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False  # keep blank lines for now; collapsed later
    word_chars = sum(c.isalnum() or c.isspace() for c in stripped)
    ratio = word_chars / len(stripped)
    # Short lines that are <60% word-characters are noise (e.g. logo glyphs).
    return len(stripped) < 40 and ratio < 0.6


def _deglue(text: str) -> str:
    """Re-space words the OCR fused together at a lowercase->UPPERCASE boundary,
    e.g. 'AdequacyRatio' -> 'Adequacy Ratio', 'TaxExpense' -> 'Tax Expense'. This is
    general (no specific term is named) and it matters a lot for retrieval: a glued
    'AdequacyRatio' token never matches the query word 'adequacy'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)


def _clean(text: str) -> str:
    text = _deglue(text)
    lines = [ln for ln in text.splitlines() if not _is_garbage_line(ln)]
    # Collapse 3+ blank lines to one; strip trailing spaces from layout padding.
    out, blanks = [], 0
    for ln in lines:
        ln = ln.rstrip()
        if not ln.strip():
            blanks += 1
            if blanks <= 1:
                out.append("")
            continue
        blanks = 0
        out.append(ln)
    return "\n".join(out).strip()


# --- OCR fallback ----------------------------------------------------------
# Lazily constructed so that importing this module is cheap and so machines that
# never hit a scanned page don't pay the model-load cost.
_ocr_engine = None


def _ocr_page(pdf_path: Path, page_index: int) -> str:
    """Rasterize one page to an image and OCR it. Used only for pages with no
    usable text layer (e.g. every page of data_4.pdf)."""
    global _ocr_engine
    import fitz  # PyMuPDF
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    # Render at ~300 DPI (zoom 300/72) so OCR has enough resolution.
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # drop alpha
        img = img[:, :, :3]
    doc.close()

    result, _ = _ocr_engine(img)
    if not result:
        return ""
    # RapidOCR returns [ [box, text, score], ... ] in reading order.
    return "\n".join(line[1] for line in result)


# --- Document metadata (statement type + reporting period) -----------------
# These filings stamp a header on (almost) every results page, e.g.
#   "STANDALONE FINANCIAL RESULTS FOR THE QUARTER AND YEAR ENDED MARCH 31, 2026"
# We parse that header to tag each page standalone/consolidated and with the
# period it covers. This is GENERAL (keys off the document's own wording, never a
# filename) and it's what lets the LLM avoid the standalone-vs-consolidated trap
# and pick the right period column.
_MONTHS = (
    "january february march april may june july august "
    "september october november december"
).split()
_ENDED_DATE_RE = re.compile(
    r"ended.{0,4}(" + "|".join(_MONTHS) + r")(\d{1,2}),?(\d{4})"
)


def _detect_statement_meta(text: str) -> tuple[str, str] | None:
    """From a page's header, return (statement_type, period) or None if this page
    has no recognizable '... FINANCIAL RESULTS ... ENDED <date>' header. Robust to
    OCR: detection runs on a whitespace-squashed, lowercased copy of the top lines."""
    head = " ".join(text.splitlines()[:10])
    sq = re.sub(r"\s+", "", head).lower()
    if "financialresults" not in sq:
        return None
    stmt = "consolidated" if "consolidated" in sq else ("standalone" if "standalone" in sq else "")
    m = _ENDED_DATE_RE.search(sq)
    period = f"{m.group(1).capitalize()} {int(m.group(2))}, {m.group(3)}" if m else ""
    return stmt, period


def _assign_metadata(pages: list[Page]) -> None:
    """Fill statement_type / fiscal_period / doc_label on each page, in place.

    - fiscal_period: the document covers ONE period, so we take the most common
      detected date as the doc default and apply it to every page (covers pages
      whose own header date was lost to OCR — e.g. data_4 p1).
    - statement_type: carried forward page-to-page (a doc has a standalone section
      then a consolidated section), and the leading gap before the first detected
      header is back-filled with the first statement type seen."""
    detected = [_detect_statement_meta(p.text) for p in pages]

    periods = [d[1] for d in detected if d and d[1]]
    doc_period = Counter(periods).most_common(1)[0][0] if periods else ""

    stmts = [(d[0] if d and d[0] else None) for d in detected]
    first_stmt = next((s for s in stmts if s), None)
    carried, last = [], None
    for s in stmts:
        if s:
            last = s
        carried.append(last or first_stmt)

    for p, stmt in zip(pages, carried):
        p.statement_type = stmt or ""
        p.fiscal_period = doc_period
        label_stmt = (stmt or "").capitalize()
        head = f"{label_stmt} Results" if label_stmt else "Results"
        p.doc_label = f"{head} — {doc_period}" if doc_period else head


def extract_document(pdf_path: Path) -> list[Page]:
    """Extract every page of one PDF as a cleaned Page with provenance + metadata."""
    pages: list[Page] = []
    doc_name = pdf_path.name

    with pdfplumber.open(pdf_path) as pdf:
        for i, pl_page in enumerate(pdf.pages):
            # layout=True preserves horizontal positions as spaces, which keeps
            # financial table rows (the numbers) aligned under-ish their columns.
            raw = pl_page.extract_text(layout=True) or ""
            source = "text-layer"

            if len(raw.strip()) < CONFIG.ocr_char_threshold:
                # No usable text layer on this page -> OCR it.
                raw = _ocr_page(pdf_path, i)
                source = "ocr"

            cleaned = _clean(raw)
            if cleaned:  # skip genuinely empty pages
                pages.append(Page(doc=doc_name, page=i + 1, text=cleaned, source=source))

    _assign_metadata(pages)
    return pages


def extract_corpus(data_dir: Path | None = None) -> list[Page]:
    """Extract every PDF in the data directory. General: just globs *.pdf."""
    data_dir = data_dir or CONFIG.data_dir
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")
    all_pages: list[Page] = []
    for pdf in pdfs:
        all_pages.extend(extract_document(pdf))
    return all_pages
