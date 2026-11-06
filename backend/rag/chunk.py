"""Split cleaned pages into retrieval-sized chunks, preserving provenance.

Key decisions (all explained in DECISIONS.md and the study guide):

  * Chunks NEVER cross a page boundary. The citation unit the brief asks for is
    (document, page), so a chunk that spanned two pages couldn't be cited
    honestly. One page -> one or more chunks, never the reverse.

  * We pack by the EMBEDDING MODEL'S OWN tokenizer, not by characters. bge-small
    truncates at 512 tokens; if a chunk overflows that window the tail is
    silently dropped at embed time and becomes unretrievable. Counting real
    tokens guarantees we stay inside the window.

  * We split on blank-line blocks (paragraphs / table groupings) and pack
    greedily, with a small token overlap so a fact sitting on a chunk boundary
    still appears whole in at least one chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .config import CONFIG
from .extract import Page


@dataclass
class Chunk:
    id: str           # stable id: "<doc>::p<page>::c<index>"
    doc: str
    page: int
    text: str
    source: str       # "text-layer" or "ocr", inherited from the page
    # --- metadata carried from the page (for citations + grounding) ---
    statement_type: str = ""   # "standalone" | "consolidated" | ""
    fiscal_period: str = ""    # the period the filing covers, e.g. "March 31, 2026"
    doc_label: str = ""        # human-readable citation label
    chunk_type: str = "prose"  # "table" (page-level, dated table), "row" (one table row), or "prose"
    chunk_index: int = -1      # position of this chunk within its document (set in chunk_pages)


# A dd.mm.yyyy-style reporting date (the table's column headers are made of these).
_DATE_TOKEN = re.compile(r"\b\d{1,2}[.\-/]\d{2}[.\-/]\d{4}\b")
# The period-grouping words above the date row ("Quarter ended", "Year ended", ...).
_PERIOD_WORDS = re.compile(r"(quarter|half[\s-]*year|nine[\s-]*months|year|period)\s*ended", re.I)
_PERIOD_WORDS_SQUASHED = re.compile(r"(quarter|halfyear|ninemonths|year|period)ended", re.I)


def _detect_period_header(text: str) -> str | None:
    """Return the table's column-header block (period-grouping words + the reporting
    dates), or None if this page is not a period-dated table.

    This is the heart of the fix: chunking otherwise splits these header rows away
    from the value rows below them, so a retrieved chunk like "Net Profit ...
    74671.29 67347.36" loses all sense of WHICH column is which period. We detect
    the header once per page and re-attach it to every chunk that lost it.

    Must handle two very different layouts:
      * digital PDFs  -> the dates sit on ONE line: "Particulars 30.09.2025 ... 31.03.2025"
      * OCR'd scans   -> each date / word is its OWN line: "Quarterended" / "31.03.2026" / ...
    So we collect a contiguous run of header lines from the top of the page (lines
    that are a period word or contain a reporting date), and stop once two ordinary
    data rows go by. We require >=2 dates total (or a period word) so a stray date
    in a prose note doesn't get mistaken for a table header."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header: list[str] = []
    started = False
    gap = 0
    total_dates = 0
    has_period = False
    for line in lines[:30]:
        squashed = line.replace(" ", "")
        period = bool(_PERIOD_WORDS.search(line)) or bool(_PERIOD_WORDS_SQUASHED.search(squashed))
        n_dates = len(_DATE_TOKEN.findall(line))
        if period or n_dates >= 1:
            header.append(" ".join(line.split()))  # collapse layout padding
            started, gap = True, 0
            total_dates += n_dates
            has_period = has_period or period
        elif started:
            gap += 1
            if gap >= 2:  # two ordinary rows in a row -> we're past the header
                break
    if not header or (total_dates < 2 and not has_period):
        return None
    return " ".join(header)


# --- Row-level chunks --------------------------------------------------------
# Page-sized table chunks are dense blobs of OCR'd numbers: the "Net Profit" row and
# the "Capital Adequacy Ratio" row share one ~1000-char chunk with ten other rows, so
# its embedding is a blur and it ranked 57th-153rd of 224 for those questions (BM25
# fared no better). We therefore ALSO index every table row on its own, with the
# document label + column header attached — a small, focused target. The page-level
# chunks stay: they give the LLM surrounding context.
_NUM = re.compile(r"^[\(\[]?-?(\d[\d,]*(\.\d+)?%?|\d{1,2}[.\-/]\d{2}[.\-/]\d{4})[\)\]]?$")
_DATE = re.compile(r"^\d{1,2}[.\-/]\d{2}[.\-/]\d{4}$")
_ROW_NO = re.compile(r"^\d{1,2}$")                 # a bare row number like "13"
_PLACEHOLDERS = frozenset({"nil", "nit", "iin", "-", "—", "–"})  # empty cells (incl. OCR variants)
_MIN_VALUES = 3                                      # a value run needs >=3 numbers (tables have 3-5 columns)
_LABEL_MAX_WORDS = 40                                # keep the words nearest the values


def split_row_parts(text: str) -> list[tuple[list[str], list[str]]]:
    """Like split_rows but returns (label_words, value_tokens) per row, so callers can
    pair each value with its column."""
    tokens = text.split()
    n = len(tokens)
    rows: list[tuple[list[str], list[str]]] = []
    label: list[str] = []
    i = 0
    while i < n:
        if not _NUM.match(tokens[i]):
            # A run of >=3 empty-cell markers ("Nil Nit IIN IIN Nil") is a valueless row
            # of its own; without this its label would be glued onto the NEXT row.
            k = i
            while k < n and tokens[k].lower() in _PLACEHOLDERS:
                k += 1
            if k - i >= _MIN_VALUES:
                rows.append((label[-_LABEL_MAX_WORDS:], tokens[i:k]))
                label = []
                i = k
                continue
            label.append(tokens[i])
            i += 1
            continue
        j = i
        while j < n and (_NUM.match(tokens[j]) or (j > i and tokens[j].lower() in _PLACEHOLDERS)):
            j += 1
        run = tokens[i:j]
        numeric = [t for t in run if _NUM.match(t)]
        if len(numeric) < _MIN_VALUES:
            label.extend(run)                       # e.g. a row number "12" or "(Refer note 18)"
            i = j
            continue
        # A trailing bare integer followed by words is the NEXT row's number ("... 67347.36 13 Extraordinary...").
        if j < n and len(run) > _MIN_VALUES and _ROW_NO.match(run[-1]):
            j -= 1
            run = run[:-1]
            numeric = [t for t in run if _NUM.match(t)]
        if not all(_DATE.match(t) for t in numeric):     # skip the column-header date row
            rows.append((label[-_LABEL_MAX_WORDS:], run))
        label = []
        i = j
    return rows


def split_rows(text: str) -> list[str]:
    """Split a page's text into table rows: `<label words> <run of >=3 numbers>`.

    Works on the WHOLE page's flattened token stream, so it copes with how OCR
    delivers tables — several rows glued onto one line, and rows cut in two by a
    blank line. The header's date row (all dates) is dropped; a long preceding
    label is trimmed to its last words."""
    return [" ".join(label + values) for label, values in split_row_parts(text)]


# --- Column labels ------------------------------------------------------------------
# Indian results tables: "Quarter ended" = 3 columns (current, previous quarter, year-ago
# quarter); each further group ("Half year ended" / "Nine months ended" / "Year ended") =
# 2 columns (current, previous year) and, if a third group exists, 1. We pair every value
# with its column so the LLM doesn't have to infer which bare number is which. Dates come
# from the filing's detected period, NOT the OCR'd header (real OCR misread them).
_GROUP_RE = re.compile(r"(halfyear|ninemonths|quarter|year)ended")
_GROUP_TITLE = {"quarter": "Quarter ended", "halfyear": "Half year ended",
                "ninemonths": "Nine months ended", "year": "Year ended"}
_GROUP_SIZES = (3, 2, 1)
_GROUP_OFFSETS = {3: (0, 3, 12), 2: (0, 12), 1: (0,)}      # months back from the filing period
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"]
_PERIOD_TEXT = re.compile(r"^(" + "|".join(_MONTH_NAMES) + r")\s+(\d{1,2}),\s*(\d{4})$")


def _shift_period(period: str, months_back: int) -> str | None:
    """'March 31, 2026' shifted back N months -> 'December 31, 2025' (month-end)."""
    import calendar

    m = _PERIOD_TEXT.match((period or "").strip())
    if not m:
        return None
    total = int(m.group(3)) * 12 + _MONTH_NAMES.index(m.group(1)) - months_back
    year, month0 = divmod(total, 12)
    return f"{_MONTH_NAMES[month0]} {calendar.monthrange(year, month0 + 1)[1]}, {year}"


def column_labels(header: str | None, n_values: int, fiscal_period: str) -> list[str] | None:
    """Labels like 'Year ended March 31, 2026' for a row's `n_values` columns, or None
    when the layout can't be determined with confidence (never guesses)."""
    if not header or not _PERIOD_TEXT.match((fiscal_period or "").strip()):
        return None
    groups = _GROUP_RE.findall(re.sub(r"\s+", "", header).lower())
    if not groups or len(groups) > len(_GROUP_SIZES):
        return None
    sizes = _GROUP_SIZES[: len(groups)]
    if sum(sizes) != n_values:
        return None
    labels: list[str] = []
    for group, size in zip(groups, sizes):
        for back in _GROUP_OFFSETS[size]:
            labels.append(f"{_GROUP_TITLE[group]} {_shift_period(fiscal_period, back)}")
    return labels


_COL_SEP = " | "   # separates a rendered row's label from its labelled values


def embedding_text(chunk: Chunk) -> str:
    """The text we EMBED for a chunk. Page-level chunks: exactly their text. Row chunks:
    only the row's label words — not the doc-label/header prefix (identical on every
    chunk of a filing, so it swamps the embedding) and not the numbers (they carry no
    semantics, only noise). The full text is still what's stored, shown and BM25-indexed."""
    if chunk.chunk_type != "row":
        return chunk.text
    row = chunk.text.rsplit("\n", 1)[-1].split(_COL_SEP, 1)[0]   # rendered rows: label part only
    words = [t for t in row.split() if not _NUM.match(t) and t.lower() not in _PLACEHOLDERS]
    return " ".join(words) or row


@lru_cache(maxsize=1)
def _tokenizer():
    """The embedding model's tokenizer (weights NOT loaded — just the tokenizer)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(CONFIG.embedding_model)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _blocks(text: str) -> list[str]:
    """Split page text into blocks on blank lines; keep each block intact."""
    blocks, cur = [], []
    for line in text.splitlines():
        if line.strip():
            cur.append(line)
        elif cur:
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _overlap_tail(text: str, max_tokens: int) -> str:
    """Return the last <=max_tokens worth of `text`, split on word boundaries."""
    if max_tokens <= 0:
        return ""
    words = text.split()
    tail = []
    for w in reversed(words):
        tail.insert(0, w)
        if count_tokens(" ".join(tail)) > max_tokens:
            tail.pop(0)
            break
    return " ".join(tail)


def _split_by_tokens(words: list[str], max_tok: int) -> list[str]:
    """Greedily pack words into pieces of <= max_tok tokens. Binary-searches the
    largest fitting prefix, so tokenizer calls are O(pieces * log n), not one per
    word. A single word bigger than the window still becomes its own piece (never
    an empty one)."""
    pieces: list[str] = []
    i = 0
    while i < len(words):
        lo, hi = 1, len(words) - i  # a piece always takes at least one word
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(" ".join(words[i : i + mid])) <= max_tok:
                lo = mid
            else:
                hi = mid - 1
        pieces.append(" ".join(words[i : i + lo]))
        i += lo
    return pieces


def chunk_page(page: Page) -> list[Chunk]:
    max_tok = CONFIG.chunk_max_tokens
    overlap_tok = CONFIG.chunk_overlap_tokens

    chunks: list[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""

    for block in _blocks(page.text):
        # A single block bigger than the window is split by tokens directly.
        if count_tokens(block) > max_tok:
            flush()
            pieces = _split_by_tokens(block.split(), max_tok)
            chunks.extend(pieces[:-1])
            cur = pieces[-1]  # keep filling the last piece with following blocks
            continue

        trial = (cur + "\n\n" + block).strip() if cur else block
        if count_tokens(trial) > max_tok:
            # Start a new chunk, seeded with the overlap tail of the last one.
            flush()
            tail = _overlap_tail(chunks[-1], overlap_tok) if chunks else ""
            cur = (tail + "\n\n" + block).strip() if tail else block
        else:
            cur = trial
    flush()

    # Re-attach the table's period/column header to every chunk that lost it.
    # A chunk that already contains the date row (usually the first one) is left
    # alone so we don't duplicate it.
    header = _detect_period_header(page.text)
    chunk_type = "table" if header else "prose"

    out: list[Chunk] = []
    for idx, body in enumerate(chunks):
        parts: list[str] = []
        # The label puts the statement type + period (as real words, incl. the
        # month name) into the indexed text, so a question like "...March 31, 2026"
        # can match the right filing even though the table itself dates columns
        # numerically ("31.03.2026").
        if page.doc_label:
            parts.append(page.doc_label)
        # Re-attach the period/column header to chunks that lost it (see above).
        if header and len(_DATE_TOKEN.findall(body)) < 2:
            parts.append(header)
        parts.append(body)
        text = "\n".join(parts)
        out.append(
            Chunk(
                id=f"{page.doc}::p{page.page}::c{idx}",
                doc=page.doc,
                page=page.page,
                text=text,
                source=page.source,
                statement_type=page.statement_type,
                fiscal_period=page.fiscal_period,
                doc_label=page.doc_label,
                chunk_type=chunk_type,
            )
        )

    # Row-level chunks (see split_rows). Same provenance as the page; header re-attached
    # so the numbers still line up with their period columns.
    if CONFIG.row_chunks:
        parts_list = split_row_parts(page.text)
        if len(parts_list) >= 2:
            for r_idx, (label, values) in enumerate(parts_list):
                cols = column_labels(header, len(values), page.fiscal_period)
                parts = [page.doc_label] if page.doc_label else []
                if cols:   # self-describing: each value next to its column; the OCR header is redundant
                    parts.append(_COL_SEP.join([" ".join(label)] + [f"{c}: {v}" for c, v in zip(cols, values)]))
                else:      # fall back to the raw row + the (re-attached) period header
                    if header:
                        parts.append(header)
                    parts.append(" ".join(label + values))
                out.append(
                    Chunk(
                        id=f"{page.doc}::p{page.page}::r{r_idx}",
                        doc=page.doc,
                        page=page.page,
                        text="\n".join(parts),
                        source=page.source,
                        statement_type=page.statement_type,
                        fiscal_period=page.fiscal_period,
                        doc_label=page.doc_label,
                        chunk_type="row",
                    )
                )
    return out


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    out: list[Chunk] = []
    per_doc: dict[str, int] = {}
    for p in pages:
        for c in chunk_page(p):
            c.chunk_index = per_doc.get(c.doc, 0)  # position within the document
            per_doc[c.doc] = c.chunk_index + 1
            out.append(c)
    return out
