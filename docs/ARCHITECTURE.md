# Architecture — Ask-the-Docs

> A grounded RAG Q&A assistant: retrieve the right passages, answer **only** from
> them, cite them by **document + page**, and **refuse** when they don't contain
> the answer.

This document maps the system against **Section 3 (Required Features — the core)**
of the brief, because the brief is explicit: *"Do **not** start [the stretch
items] until Section 3 works end-to-end."* So the goal of this doc is to prove,
honestly, that Section 3 is solid — the flow, the design decisions, the tradeoffs,
what works, and what doesn't — **before** touching Section 4.

---

## 1. The two flows at a glance

The system is two halves that share exactly **one** thing: the embedding model.
Chunks and questions are embedded by the *same* model into the *same* vector
space — that symmetry is the only reason similarity search means anything.

```
INGEST  (offline, run once)                     ASK  (per question, live)
──────────────────────────                      ─────────────────────────
data/*.pdf                                       POST /ask { question }
   │  extract.py  — text per page                    │  store.search()
   │   └─ OCR fallback if no text layer              │   ├─ embed.embed_query()
   │  _clean()   — drop garbled glyph lines          │   └─ cosine over NumPy matrix
   │  chunk.py   — page-aware, ≤350 tokens           │
   │  embed.py   — bge-small, local, normalized      ▼  pipeline.ask()
   ▼                                                 GATE 1: score floor (pre-LLM)
storage/index/                                       │   best sim < 0.30  → refuse
   ├─ embeddings.npy   (N×384, float32, L2-norm)     │  build numbered context block
   └─ chunks.json      (text + {doc,page,source})    │  generate.py → Groq LLM (JSON, temp 0)
                                                      │  GATE 2: LLM says "not_covered" → refuse
        one-way: ASK only READS the index            │  map used_passages → citations
                                                      ▼
                                          { answer, sources[], refused }
```

**Key property:** the API never ingests on the fly. Ingestion is a deliberate,
one-shot offline step (`python ingest.py`); the live API only *reads* the index.
This keeps requests fast and makes the index reproducible.

---

## 2. Component map (where each Section-3 requirement lives)

| Brief requirement | Module | What it does |
|---|---|---|
| **3.1** parse + chunk | `backend/rag/extract.py` | PDF → clean per-page text (+ OCR fallback) |
| **3.1** provenance | `backend/rag/chunk.py` | per-page chunks carrying `{doc, page, source}` |
| **3.1** embed + store | `backend/rag/embed.py`, `store.py` | local bge-small vectors; exact cosine over NumPy |
| **3.1** ingestion CLI | `backend/ingest.py` | extract → chunk → embed → store, one command |
| **3.2** retrieval + grounding | `backend/rag/pipeline.py` | retrieve → 2 gates → prompt → parse → cite |
| **3.2** LLM isolation | `backend/rag/generate.py` | the **only** place that talks to a model |
| **3.2** HTTP API | `backend/app.py` | `GET /health`, `POST /ask` |
| **3.3** frontend | `frontend/src/App.jsx` + components | ask box, answer, sources, loading/error |
| all tunables | `backend/rag/config.py` | every knob, loaded from `.env` |
| retrieval inspector | `backend/query.py` | eyeball raw retrieval (debug, not in app) |

---

## 3. Design decisions & tradeoffs (per requirement)

### 3.1 — Ingestion → searchable index

**Extraction (`extract.py`).** The four PDFs are *not* uniform (verified by
inspection): `data_2` is a clean digital PDF, `data_1`/`data_3` are scans with a
usable hidden text layer, `data_4` is a pure scan with **zero** text. So the
pipeline is **general, not per-file**:
- `pdfplumber.extract_text(layout=True)` keeps table columns roughly aligned via
  spaces, which matters for financial rows.
- If a page yields < 50 chars (`ocr_char_threshold`), it's rasterized at 300 DPI
  and OCR'd with **RapidOCR** (pure pip — no system Tesseract install).
- `_clean()` drops "garbage" lines (short + mostly non-word chars) by a *ratio
  test*, never by matching specific strings. The brief forbids special-casing.
- **Tradeoff:** a generic cleanup filter occasionally keeps minor OCR noise and
  can't rescue scrambled multi-row table *headers*. We accept that and rely on
  citations as the safety net, rather than building a fragile table parser.

**Chunking (`chunk.py`).**
- **Chunks never cross a page boundary** — the citation unit is `(document,
  page)`, so a cross-page chunk couldn't be cited honestly.
- Packed by the **embedding model's own tokenizer** (≤350 tokens, ~60 overlap),
  not by characters — bge-small truncates at 512 tokens, so a char-based chunk
  could silently lose its tail at embed time and become unretrievable.
- **Tradeoff:** 350 leaves headroom but produces a couple of tiny chunks; we
  chose simplicity over a minimum-size merge pass.

**Embeddings (`embed.py`).** `BAAI/bge-small-en-v1.5`, local, ~130 MB, CPU-
friendly, no API key. Uses the bge **query-side prefix** only on the question —
bge is trained for that asymmetry and it measurably improves recall.

**Vector store (`store.py`).** Exact cosine over a NumPy matrix — **not**
Chroma/FAISS/HNSW. This is the decision most likely to be probed:
- At ~224 chunks, one matrix multiply is sub-millisecond **and exact** (no ANN
  recall loss). ANN indexes exist to make search sublinear at *millions* of
  vectors — we don't have that problem.
- **Zero native dependencies.** Chroma's `hnswlib` needs a C++ toolchain and
  failed to build on Windows; NumPy removes that entire class of setup failure
  and keeps "clone and run" true (a Section-5 hard rule).
- **Honest boundary:** at ~100k+ chunks I'd switch to FAISS.

### 3.2 — Retrieval + grounded answer

The brief calls grounding & honesty the **#1 graded criterion**. The core design
is **two grounding gates**, because neither alone is enough:

1. **Gate 1 — score floor (pre-LLM, cheap).** If nothing retrieved clears
   `min_similarity` (0.30), refuse *without* spending an LLM call. Catches the
   catastrophic "nothing relevant retrieved" case.
2. **Gate 2 — the LLM's own judgment.** The model answers in strict JSON
   (`{not_covered, answer, used_passages}`) at `temperature=0`, instructed to use
   *only* the numbered passages. If it flags `not_covered` (or returns an empty
   answer), we refuse. Catches "we retrieved something, but it doesn't actually
   answer the question."

**Why two, measured live:** "CEO of Microsoft" (unanswerable) scored 0.499 —
*above* the floor, so Gate 1 lets it through; Gate 2 refuses it. Pure gibberish
scored 0.56 — *higher* than some real unanswerable questions. So similarity alone
is a **weak** refusal signal; the LLM is the *primary* grounding judge and the
floor is just a cheap pre-filter.

Other choices worth defending:
- **Fail safe:** unparseable LLM output → refuse, never guess.
- **Provider isolation:** swapping Groq → OpenAI → local Ollama is a one-function
  change in `generate.py`; nothing else knows who answers.
- **Clean errors:** dead key / rate limit → `LLMError` → HTTP **502** with a
  helpful message; empty question → 422 (Pydantic).
- **`generate` is injectable** — `ask(..., generate=fake)` lets tests exercise the
  whole pipeline with no key and no network.

### 3.3 — Frontend (React + Vite)

- Single API call in `api.js` to `/api/ask`; Vite **proxies** `/api/*` to the
  backend, so the browser sees same-origin (no CORS) and code hardcodes no host.
- **Safe rendering:** answer + snippets are rendered as `{text}` in JSX (React
  escapes); no `dangerouslySetInnerHTML`. Satisfies the brief's "render safely".
- Distinct **loading / error / refused / answered** states; Ask is disabled while
  loading or empty (no duplicate/empty requests).
- **Tradeoff:** plain CSS, no UI framework — the brief penalizes framework
  sprawl and rewards correctness over polish.

### 3.4 — Self-evaluation

Status: **not yet written as a deliverable.** See "what's not done" below — this
is the one part of Section 3 that is incomplete.

---

## 4. What's working fine ✅

- **End-to-end happy path.** PDF corpus → index → grounded answer with citations,
  browser → proxy → FastAPI → Groq → JSON, all verified.
- **General ingestion.** Verified routing: `{'text-layer': 46, 'ocr': 23}` — all
  23 pages of the zero-text `data_4` correctly fell back to OCR; no file is named
  in code, so a swapped-in document would work.
- **Honest refusal.** Both gates fire correctly on unanswerable questions and
  gibberish; the system declines instead of inventing.
- **Citations.** Every answer maps back to real `(doc, page, snippet)` the user
  can open and verify; if the model forgets to cite, we fall back to the top hit.
- **Robust retrieval.** Exact cosine, sub-millisecond, no native deps, runs on a
  clean Windows machine.
- **Clean failure modes.** Missing key, network down, malformed model output, and
  empty input all produce clear, non-crashing responses.

## 5. What's NOT working fine yet ⚠️ (be upfront — the brief rewards this)

| Issue | Why | Mitigation today | Real fix |
|---|---|---|---|
| **Period-specific questions hit the wrong filing** | Scrambled multi-row date headers + no per-document report-date metadata; the model can't reliably map a number to a period column | The citation shows *which* doc it used, so the user can catch it | Extract per-doc reporting-date metadata; hybrid retrieval + doc filters (Section 4) |
| **"First column = most recent" is an assumption** | A general RBI-format convention baked into the prompt, not a per-file rule — but may not hold for an arbitrary swapped-in doc | Documented as an assumption | Detect column→period mapping instead of assuming |
| **Residual non-determinism** | Even at `temperature=0` + `seed=42`, Groq can flip on borderline cases | Honest disclosure; don't claim perfect reproducibility | — (inherent to LLM inference) |
| **A few tiny chunks** | No minimum-size merge pass | Negligible impact at this scale | Add a min-size filter/merge |
| **OCR noise** | Lost spaces / misread letters on scanned pages (`Relurn on assels`) | Numbers come through accurately; retrieval is robust | Better OCR post-processing |
| **Self-evaluation (3.4) not authored** | Not built yet | — | Write the 8 questions + verdicts → `EVALUATION.md` |

---

## 6. Section-3 readiness checklist (do these before any Section-4 stretch)

- [x] **3.1** Ingestion → searchable index with provenance — *done & verified*
- [x] **3.2** Retrieval + grounded answer API with honest refusal — *done & verified*
- [x] **3.3** Frontend: ask box, answer, cited sources, loading/error — *done*
- [ ] **3.4** Self-evaluation: 8 questions (≥1 unanswerable, ≥1 paraphrased) with
      system answer, cited sources, and an honest verdict each → `EVALUATION.md`
- [ ] **Deliverables** the brief lists alongside Section 3: `README.md`,
      `DECISIONS.md`, `EVALUATION.md`, a few tests (chunking/provenance +
      retrieval + one API test), demo video

> **Bottom line:** the runtime core (3.1–3.3) works end-to-end and is defensible.
> The gap before "Section 3 solid" is **3.4 + the written deliverables (tests,
> DECISIONS.md, EVALUATION.md, README)**. Close those, then — and only then —
> pick *one or two* Section-4 stretch items (hybrid retrieval or filters are the
> natural ones, since they directly fix the period/wrong-filing weakness above).
```
