# DECISIONS.md

For each key decision: **what we chose**, **what we considered instead**, **why**,
and — where relevant — **what's still weak**. We were deliberate about these and
are happy to defend them; we're equally upfront about the soft spots.

---

## 1. Parsing: `pdfplumber` (layout) + OCR fallback, routed per page

**Chose:** `pdfplumber.extract_text(layout=True)` first; if a page yields < 50
characters, fall back to OCR (rasterize at 300 DPI with PyMuPDF → RapidOCR).

**Considered:** PyMuPDF text dump (jumbles table columns); Tesseract (`pytesseract`).

**Why:** The four PDFs are *not* uniform — verified by inspection:
- `data_2`: clean digital text layer.
- `data_1`, `data_3`: scanned image + a usable hidden OCR text layer (only logo
  glyphs are garbled).
- `data_4`: a pure scan ("Microsoft Print To PDF") with **zero** text layer —
  every page must be OCR'd.

So the rule is **general, not per-file**: "if a page has almost no text, OCR it."
Verified at ingest: extraction source = `{'text-layer': 46, 'ocr': 23}` — all 23
pages of `data_4` correctly fell back to OCR. `layout=True` keeps table rows
roughly column-aligned, which matters for the numbers.

**RapidOCR over Tesseract on purpose:** RapidOCR is pure-pip (ONNX), no system
install. Tesseract needs an OS-level binary — that breaks the brief's "clone and
run on a reviewer's machine" requirement on Windows. Same reasoning that pushed us
off ChromaDB (see §4).

**Weak spot:** OCR introduces noise — dropped spaces, mis-read letters (`Relurn on
assels`), and **glued words** (`AdequacyRatio`, `Quarterended`). We added a generic
`_deglue()` step (re-space at lowercase→UPPERCASE boundaries) to recover keyword
matchability, but it can't fix everything. Numbers come through accurately; we cite
the page so a human can verify.

---

## 2. Cleaning: generic "garbage line" filter, no per-file rules

**Chose:** Drop short lines that are mostly non-word characters (a ratio test) to
remove garbled logo/header glyphs. De-glue camel-cased OCR runs.

**Considered:** Hand-matching the specific junk strings.

**Why:** The brief forbids special-casing the provided documents. A ratio test is
general and would handle a brand-new scanned document the reviewer swaps in.

---

## 3. Chunking: page-bounded, ≤350 tokens, header re-attached, doc-label-prefixed

**Chose:**
- **Never cross a page boundary** — the citation unit is `(document, page)`, so a
  chunk spanning two pages couldn't be cited honestly.
- Pack to **≤350 tokens** measured by the **embedding model's own tokenizer**
  (not characters), with ~60-token overlap.
- **Re-attach the table's period/column header** (e.g. `Quarter ended / Year
  ended / 31.03.2026 ...`) to every chunk that lost it during splitting.
- **Prepend a human `doc_label`** (statement type + period, e.g. "Standalone
  Results — March 31, 2026") to each chunk's indexed text.

**Considered:** char-based chunking; one-chunk-per-table; bigger chunks.

**Why:**
- bge-small truncates at 512 tokens — a char-based chunk could silently lose its
  tail at embed time and become unretrievable. Counting real tokens guarantees we
  stay inside the window (observed max ≈ 430 incl. the attached header).
- A whole financial table is ~1,400 tokens — far over the window — so it *must*
  be split. But naive splitting separated the period headers from the value rows,
  so a retrieved "Net Profit ... 74671.29" chunk lost all sense of which column is
  which year. Re-attaching the header fixes that (and improves retrievability).
- The doc-label prefix puts the month name and statement type (as real words)
  into the indexed text, so a question like "...as of March 31, 2026" can match a
  filing whose table dates columns numerically (`31.03.2026`).

**Metadata we keep per chunk:** `doc`, `page`, `source` (text-layer/ocr),
`statement_type` (standalone/consolidated, derived from the page header and
carried forward across a document's sections), `fiscal_period` (the period the
filing covers), `doc_label`, `chunk_type` (table/prose), `chunk_index` (position
within the document, used to present context in reading order). All derived from
the documents' own header text — never from filenames.

**Weak spot — the main one:** the dense "Analytical Ratios" block (CAR, EPS, NPA,
RoA, net worth) is so number-heavy that the discriminating phrase (e.g. "Capital
Adequacy Ratio") gets diluted, and adjacent ratios sometimes land in different
chunks. So value-row lookups like CAR or net profit rank lower than prose, and
occasionally fall outside the top-k. Measured with the token-free harness:
~10/17 value-row questions land the answer chunk in the top-8. This affects **all**
documents (it's a dense-table problem, not a `data_4`/OCR problem — `data_4`
actually ranks *best* of the four). The honest next step is finer-grained chunking
of the ratios sub-section; we documented it rather than shipping a rushed change.

---

## 4. Embeddings + vector store: local bge-small, exact cosine over NumPy

**Chose:** `BAAI/bge-small-en-v1.5` (local, ~130 MB, 512-token window), with the
bge query-side instruction prefix on questions only. Store = an L2-normalized
NumPy matrix; search = one matrix multiply (exact cosine).

**Considered:** OpenAI `text-embedding-3-small` (paid, needs a key); Chroma / FAISS
/ HNSW.

**Why:**
- Local + free keeps the brief's "cheap to run" promise; no key, no per-query cost.
- At ~224 chunks, a matrix multiply is sub-millisecond **and exact** — approximate
  NN indexes (HNSW) exist to go sublinear at *millions* of vectors, a problem we
  don't have, and they trade recall for speed. NumPy also has **zero native
  dependencies**: we tried Chroma first and its `hnswlib` failed to compile on
  Windows — a reviewer would hit the same wall. **Honest boundary:** at ~100k+
  chunks we'd switch to FAISS.

---

## 5. Retrieval: hybrid (vector + BM25) fused by RRF, + recency tiebreak

**Chose:** Fuse the vector ranking with a BM25 keyword ranking via Reciprocal Rank
Fusion. Strip stopwords/bare-numbers from the BM25 *query* only. Add a small
recency bonus when the question names no period.

**Considered:** vector-only; keyword-only; weighted score blending.

**Why:** Pure vector search ranks chatty prose that echoes the question's words
*above* the terse numeric row that actually answers it (measured: a value row fell
to rank #118 under pure vector). BM25 rewards the literal rare phrase ("adequacy")
in the value row. RRF lets a chunk strong in *either* signal surface. Stopword
filtering stops "...as of March 31, 2026" from matching every page header. The
recency bonus makes a generic "what was total income?" prefer the latest filing;
an explicit year/month disables it so period-specific questions still match exactly.

---

## 6. Cross-encoder reranker — **tried, then dropped** (honest stretch story)

**Tried:** a two-stage retriever — gather ~25 candidates, then re-score each
`(question, chunk)` pair jointly with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Dropped it (default off).** Why: the cross-encoder rewards text that "reads like
an answer," so it reordered terse numeric value rows *below* fluent prose decoys —
e.g. it ranked an OFS "net **gain** on sale of shares ₹9,179 cr" note above the
real "Net Profit for the period" row, and the LLM then answered with the wrong
figure. We even tried a numeric-aware rank-fusion variant; an A/B on the token-free
harness showed it gave **no recall improvement** (10/17 either way) while adding an
80 MB model, latency, and that failure mode. So we removed it from the default path
(still toggleable via `USE_RERANKER=true`). **Lesson:** a reranker is not free
upside — on dense numeric tables it can actively mislead.

---

## 7. Grounding & refusal: two gates

**Chose:** (1) a pre-LLM **similarity floor** (`MIN_SIMILARITY=0.30`) — if nothing
retrieved clears it, refuse without an LLM call; (2) the **LLM's own judgment** —
it answers in strict JSON and flags `not_covered` when the passages don't contain
the answer. Refuse if either fires, or if the JSON is unparseable (fail safe).

**Considered:** a single similarity threshold.

**Why:** Neither alone is enough. Measured: "CEO of Microsoft" scored 0.499 — above
the floor — so the floor alone would leak it; the LLM catches it. Gibberish once
scored *higher* than a real unanswerable question, so similarity is a weak refusal
signal. The LLM is the primary judge; the floor is a cheap pre-filter for the
"nothing retrieved" case. The prompt is explicitly tuned **not to over-refuse** on
messy OCR tables (answer if any passage holds a matching figure; refuse only when
no passage mentions the metric), after we saw it decline valid questions.

**Weak spot:** even at `temperature=0` + `seed=42`, the LLM isn't perfectly
deterministic on borderline questions; and an over-soft prompt can trade a false
refusal for a confidently-wrong answer on the value-row questions (§3). We favor
refusing on genuine absence and document where it can still mis-answer.

---

## 8. Frontend: React + Vite, plain CSS, safe rendering

**Chose:** a single ask box, answer + per-source cards (doc label, page, % match,
snippet), explicit loading/error/refused states. All model/document text rendered
as escaped JSX text (no `dangerouslySetInnerHTML`). A Vite proxy forwards `/api/*`
to the backend (no CORS, no hardcoded host).

**Why:** The brief penalizes framework sprawl and rewards correctness. Escaped
rendering satisfies "render the model's text safely."

---

## 9. Known limitations (the honest list)

1. **Dense value-row retrieval** (§3) — CAR / net-profit rows can rank outside the
   top-k; ~10/17 value-row questions retrieve the exact answer chunk. Real fix:
   finer chunking of the ratios block. Affects all docs, not just the OCR'd one.
2. **Standalone vs consolidated** — both appear in the same filing; the metadata
   label helps the LLM distinguish them, but a buried value row can still cause a
   wrong-statement answer.
3. **OCR noise** on `data_4` — numbers are accurate, surrounding words can be
   mangled; `_deglue` mitigates but doesn't eliminate it.
4. **LLM non-determinism** and the **Groq free daily token cap** (~100k/day) —
   account limits, not bugs; the API degrades cleanly to a 502.
