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
    chunk_type: str = "prose"  # "table" if the page carries a period-dated table, else "prose"
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
