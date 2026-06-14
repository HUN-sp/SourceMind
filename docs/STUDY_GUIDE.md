# Study Guide — Ask-the-Docs (interview prep)

This document explains the system so you can defend every decision live. It grows
one section per phase. Read it top-to-bottom and you can answer "walk me through
your architecture" and "why did you do X instead of Y" for every part.

---

## The one-sentence summary (memorize this)

> Retrieve the right passages, answer **only** from them, cite them with document
> and page, and **refuse** when they don't contain the answer.

That single sentence *is* the assignment (it mirrors Section 8's grading order:
grounding & honesty → retrieval quality → citations → engineering → frontend).

---

## The whole pipeline at a glance

```
INGEST (offline, run once)                 ASK (per question, live)
─────────────────────────                  ─────────────────────────
PDFs                                        user question
  │ extract text per page                     │ embed question (same model)
  │  └─ OCR fallback if no text layer          │ cosine search over chunks
  │ clean                                      │ grounding gate (score floor)
  │ chunk (page-aware, ≤350 tok)               │ LLM answers ONLY from chunks
  │ embed (local, bge-small)                   │   └─ or says "not covered"
  ▼                                            ▼
NumPy index (embeddings.npy + chunks.json)  { answer, sources[], refused }
```

The two halves share **one thing**: the embedding model. Chunks and questions are
embedded by the *same* model into the *same* vector space — that's the only reason
similarity search means anything. If you remember one mechanical fact, remember
that symmetry.

---

# PHASE 1 — Ingestion → searchable index  ✅

**Goal:** turn the PDFs into something we can search, never losing track of which
document and page each piece of text came from (that's what makes honest
citations possible later).

### Files and what each does
| File | Job |
|---|---|
| `rag/config.py` | All tunables in one place, loaded from `.env`. |
| `rag/extract.py` | PDF → clean per-page text, with an OCR fallback. |
| `rag/chunk.py` | Per-page text → retrieval-sized chunks, with provenance. |
| `rag/embed.py` | Text → vectors (local model, no API key). |
| `rag/store.py` | Store/search vectors via exact cosine over NumPy. |
| `ingest.py` | The CLI that runs extract → chunk → embed → store. |
| `query.py` | Manual inspector to eyeball retrieval (not used by the app). |

### The corpus is NOT uniform — this drove the whole design
I inspected the four PDFs before writing any code (you should be ready to say this
— it shows you didn't just prompt an AI for a generic RAG demo):

| Doc | What it really is | How we read it |
|---|---|---|
| data_2 | Clean digital PDF (embedded font) | Text layer, reliable |
| data_1, data_3 | Scanned image + a *hidden* OCR text layer; only the logo glyphs are garbled | Text layer + a generic cleanup filter |
| data_4 | **Pure scan** ("Microsoft Print To PDF"), **zero** text, one image per page | **Must OCR every page** |

**Verified at ingest time:** extraction source = `{'text-layer': 46, 'ocr': 23}`.
All 23 of data_4's pages correctly fell back to OCR; the other 46 used their text
layer. That number is your proof the routing works.

### Key decisions (the "why", for the interview)

**1. Extraction: `pdfplumber.extract_text(layout=True)` + OCR fallback.**
- `layout=True` keeps horizontal positions as spaces, so financial table *rows*
  (the numbers) stay aligned. Plain extraction jumbles them.
- The fallback rule is **general, not per-file**: if a page yields < 50 chars, we
  rasterize it at 300 DPI and OCR it. The same logic handles a brand-new scanned
  doc a reviewer swaps in — which the brief explicitly tests for. No file is ever
  named in the code (the brief forbids special-casing).
- OCR engine is **RapidOCR (rapidocr-onnxruntime)** — pure pip, no system Tesseract
  install. This keeps "clone and run" true on any machine.

**2. Cleaning: drop "garbage" lines generically.** Short lines that are mostly
non-word characters (the garbled `[' " l=1•1iil` header glyphs) are removed by a
ratio test — not by matching specific strings. General, not hardcoded.

**3. Chunking: page-aware, ≤350 tokens, ~60-token overlap.**
- **Never cross a page boundary.** Our citation unit is (document, page); a chunk
  spanning two pages couldn't be cited honestly. So one page → one or more chunks.
- **Counted with the embedding model's own tokenizer**, not characters. bge-small
  truncates at 512 tokens; if a chunk overflowed, its tail would be silently
  dropped at embed time and become unretrievable. 350 leaves headroom.
- **Overlap** so a fact sitting on a chunk boundary still appears whole somewhere.
- Result: 224 chunks, avg 269 tokens. (One known wrinkle: a couple of tiny chunks
  — we may add a minimum-size filter; noted honestly.)

**4. Embeddings: `BAAI/bge-small-en-v1.5` (local).** Free, no key, CPU-friendly
(~130MB). 512-token window (vs MiniLM's 256) matters because financial tables are
dense. We use the bge **query prefix** (`"Represent this sentence for searching
relevant passages:"`) only on the *question* side — bge is trained for that
asymmetry and it measurably improves recall.

**5. Vector store: exact cosine over a NumPy matrix — NOT Chroma/FAISS/HNSW.**
This is the decision most likely to be probed, so own it:
- Our corpus is ~224 chunks. Approximate-NN indexes (HNSW) exist to make search
  sublinear at *millions* of vectors. At 224, one matrix multiply is sub-
  millisecond **and exact** (no approximation recall loss).
- It has **zero native dependencies**. (We *tried* Chroma first — its `hnswlib`
  needs a C++ compiler and failed to build on Windows. A reviewer would hit the
  same wall. Switching to NumPy removed an entire class of setup failure.)
- Honest boundary: if the corpus grew to ~100k+ chunks I'd switch to FAISS.
- Persistence = two files: `embeddings.npy` (N×384, normalized) + `chunks.json`
  (text + provenance), aligned by row.

### What Phase 1 proves (verification results)
Run `python query.py "<question>"`. Observed:
- "gross NPA ratio" → data_4 **p1 (OCR)**, sim 0.72, snippet shows
  `% of Gross NPAs to Gross Advances 1.15% ...` → **verifiable on the real page**.
- "treasury segment revenue" → the three Segment-information pages, sim ~0.68.
- "capital of France" (unanswerable) → best sim only **0.49**, clearly below the
  real questions (0.66–0.75) **but above our 0.30 floor**. → This is exactly why
  Phase 2 needs a *second* grounding layer (the LLM refusing), not just a score
  threshold. Great thing to volunteer in the interview.

### Known limitations (be honest — the brief rewards this)
- OCR drops some spaces and mis-reads a few letters (`Weunderstandyourworld`,
  `Relurn on assels`). Numbers come through accurately; retrieval is robust to it.
- Multi-row table *headers* (e.g. the "Quarter ended" date row) are scrambled in
  the text layer because the source PDF interleaves them. Data rows are fine. We
  chose not to build a fragile table parser; we cite the page so the user can
  verify.

### How to run Phase 1
```bash
cd backend
python ingest.py            # build the index (OCR makes first run take a few min)
python ingest.py --stats    # just print page/chunk stats, no indexing
python query.py "what is the gross NPA ratio?"   # inspect retrieval
```

---

# PHASE 2 — Retrieval + grounded answer (backend API)  ✅

**Goal:** an HTTP API that takes a question and returns a grounded answer with
citations — or an honest refusal. This is the assignment's #1 graded criterion.

### Files
| File | Job |
|---|---|
| `rag/generate.py` | The ONLY place that talks to the LLM (Groq). One function: `generate(system, user)`. |
| `rag/pipeline.py` | The brain: retrieve → gate → prompt → parse → cite. `ask()`. |
| `app.py` | FastAPI: `GET /health`, `POST /ask`. |

### The request flow (your live walkthrough script)
A `POST /ask {question}` runs `pipeline.ask()`, which does:

1. **Retrieve** the top-k chunks (`store.search`). Each carries (doc, page).
2. **Gate #1 — score floor (pre-LLM).** If there are no chunks, or the best
   similarity < `MIN_SIMILARITY` (0.30), refuse immediately — no LLM call. Cheap.
3. **Build the prompt.** Chunks are rendered as a numbered block, each tagged
   `[n] (document: X, page: Y)`. The system prompt says: *answer ONLY from these
   passages; no outside knowledge; if they don't answer it, refuse; cite the
   passage numbers you used.* We force **JSON output** and **temperature 0**.
4. **Call the LLM** (`generate`). It returns
   `{"not_covered": bool, "answer": str, "used_passages": [int]}`.
5. **Gate #2 — the LLM's judgment.** If `not_covered` is true (or the answer is
   empty), refuse. This catches "we retrieved something, but it doesn't actually
   answer the question."
6. **Citations.** Map `used_passages` back to their chunks → that's the `sources`
   list (doc, page, snippet). If the model answered but forgot to cite, we fall
   back to the top chunk rather than show nothing.
7. Return `{answer, sources[], refused}`.

### Why TWO gates (the key insight — verified live)
Neither gate alone is enough. Measured on the real index:
- "CEO of Microsoft" (unanswerable) scored **0.499** — *above* the 0.30 floor, so
  Gate #1 passes it. Gate #2 (the LLM) catches it and refuses.
- Pure gibberish ("asdfqwer zxcv") scored **0.56** — *higher* than a real
  unanswerable question. So similarity is a **weak** refusal signal on its own.

**Conclusion to say out loud:** the LLM is the *primary* grounding judge; the
score floor is a cheap pre-filter for the catastrophic "nothing retrieved" case.
A single-threshold design (which a naive RAG demo uses) would either leak
gibberish through or reject valid paraphrased questions. We tune the floor in
Phase 4, but we never rely on it alone.

### Design choices worth defending
- **temperature 0 + JSON mode**: deterministic, parseable, no prose-wrangling.
- **Refuse on malformed output**: if the model returns junk we can't parse, we
  *refuse* rather than guess — failing safe toward honesty.
- **`generate` is injectable**: `ask(..., generate=fake)` lets tests exercise the
  whole pipeline with no API key and no network (that's how Phase 2 was verified).
- **Provider isolation**: swapping Groq → OpenAI → local Ollama is a one-function
  change in `generate.py`. Nothing else knows which model answers.
- **Error handling**: a dead key / rate limit raises `LLMError` → the API returns
  a clean **502** with a helpful message, not a stack trace. Empty question → 422.

### Debugging lessons from Phase 2 (THIS is the interview gold)
The brief says it hires for "knowing why a naive version retrieves the wrong
passage or confidently makes things up, and fixing it." Here are the real bugs we
hit and fixed — tell these stories:

**Bug 1 — confidently wrong period (grounded but imprecise).** Asked "gross NPA
ratio", the first version answered "1.33%" — a real number from the cited page,
but the *wrong period's column*. Cause: multi-period financial tables + scrambled
date headers + a period-ambiguous question. Fix: prompt the model to report
multi-period figures *with the ambiguity made explicit* rather than silently
picking one column. (Real fix for precision = the Phase 5 filters/hybrid work.)

**Bug 2 — false refusal from a retrieval miss.** "Capital adequacy ratio" was
refused even though data_1 p1 states it. Cause: the chunk with the literal value
ranked **#6**, just outside top-k=5; five "capital/ratio" look-alikes crowded it
out. This is *semantic search confusing related terms* — exactly what BM25/hybrid
(Phase 5) fixes. Interim fix: raised `TOP_K` 5 → 8 so the chunk is included.

**Bug 3 — stale `.env` overriding the config (the sneaky one).** After raising the
default `top_k` in code, `ask()` STILL used 5. Cause: `.env` is loaded *over* the
dataclass defaults, and the live `.env` had been copied from an older
`.env.example` with `TOP_K=5`. Lesson: **the running `.env` is the source of
truth, not the code default** — when behaviour doesn't match the code, check the
env first. Also a reminder that config precedence (env > default) must be
understood, not assumed.

**Bonus — non-determinism.** Even at `temperature=0`, Groq isn't 100%
deterministic on borderline questions (the same input flipped answer/refuse). We
add `seed=42` to reduce it, but the honest statement is: *LLM inference has
residual non-determinism; borderline cases can flip.* Don't claim perfect
reproducibility.

**Bug 4 — number-soup answers (caught in the UI).** Asked "gross NPA ratio", the
system returned a bare list of 8 percentages from all four filings — grounded but
useless. Cause: the prompt told it to "list values and make multi-period explicit",
which it did too literally, across all retrieved documents. Fix: prompt it to
answer like an analyst — lead with the single most-recent value, name the
document/page, mention (don't list) that earlier periods exist. Result: "The gross
NPA ratio is 1.15%, as reported in data_4.pdf, page 1."

**KNOWN LIMITATION — period-specific questions can hit the wrong filing.** "Gross
NPA ratio for the quarter ended 30 Sept 2025" answers 1.15% (from data_4, the
annual report) instead of 1.24% (from data_2, the actual Sept-2025 filing). Why:
(1) the multi-row date headers are scrambled in the source text, so the model
can't reliably map a number to a period column; (2) we don't extract each
document's reporting date as metadata, so retrieval can't prefer the right filing.
The citation is the safety net (the user sees it cited data_4, not data_2). Real
fixes: extract per-document report-date metadata; hybrid retrieval + document
filters (Phase 5). Be upfront about this — the brief rewards honesty about weak
spots over hiding them.

**The "first column = most recent" assumption.** The answer prompt tells the model
that in these statements the first numeric column is the most recent period. This
is a general convention for RBI-format periodic results, not a per-file hardcode —
but it's an *assumption* that may not hold for an arbitrary swapped-in document.
Documented as such in DECISIONS.md.

**Net design stance:** two grounding gates (score floor + LLM judgment); refuse
only on genuine absence, never on mere multi-period messiness; concise analyst-style
answers, not number dumps; `top_k=8` for recall; provider isolated behind
`generate()`; fail safe (refuse) on unparseable output.

### How to run Phase 2
```bash
cd backend
# 1. put a real free Groq key in ../.env  (GROQ_API_KEY=gsk_...)
uvicorn app:app --reload --port 8000
# then:
curl localhost:8000/health
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"what is the gross NPA ratio?"}'
```

---

# PHASE 3 — Frontend (React + Vite)  ✅

**Goal:** a clean page a non-technical user can use — type a question, read the
answer, see the sources, with sensible loading and error states.

### Files
| File | Job |
|---|---|
| `src/api.js` | The one function that calls the backend (`POST /api/ask`). |
| `src/App.jsx` | State + form + loading/error/result orchestration. |
| `src/components/AnswerPanel.jsx` | Renders answered vs refused modes. |
| `src/components/SourceCard.jsx` | One citation: doc · page · % match · snippet. |
| `src/styles.css` | Plain CSS, no UI framework (the brief penalizes sprawl). |
| `vite.config.js` | Dev server + `/api` proxy to the backend. |

### How the frontend talks to the backend (and why a proxy)
The browser calls `/api/ask`. Vite's dev server **proxies** `/api/*` to
`http://localhost:8000` (stripping `/api`). Two reasons:
- **No CORS headaches** — to the browser everything is same-origin (`:5173`).
- **Environment-agnostic code** — `api.js` never hardcodes a host; it just calls
  `/api/ask`. In production you'd point the proxy / reverse-proxy at the real API.

Verified: `GET /api/health` and `POST /api/ask` both flow browser → proxy →
backend → Groq → JSON.

### The request lifecycle in the UI (walk this for the demo)
1. User types a question, hits Ask (or Enter). `App` sets `loading=true`, clears
   any old result/error.
2. `askQuestion()` does `fetch("/api/ask", {POST, body:{question}})`.
3. **Loading state**: a spinner + "Retrieving passages…" shows.
4. On success → `AnswerPanel`:
   - **answered**: the answer text, then a `SourceCard` per cited source
     (document, page, % match, snippet) so the user can verify.
   - **refused**: an amber panel saying it wasn't found in the documents.
5. On failure → red error box. `api.js` distinguishes *network failure*
   ("is the backend running?") from a *backend error* (shows the server's
   `detail`, e.g. "GROQ_API_KEY is not set").

### Security / correctness choices
- **Safe rendering**: answer and snippets are rendered as `{text}` in JSX, which
  React **escapes**. We never use `dangerouslySetInnerHTML`, so a document (or a
  model) can't inject HTML/scripts. This satisfies the brief's "render the model's
  text safely" rule.
- **Disabled states**: Ask is disabled while loading or when the box is empty, so
  you can't fire duplicate/empty requests.
- **Example chips**: three one-click questions for a smooth demo.

### How to run Phase 3
```bash
# terminal 1 — backend (restart it so it has the latest code + .env!)
cd backend && uvicorn app:app --reload --port 8000
# terminal 2 — frontend
cd frontend && npm install && npm run dev
# open http://localhost:5173
```
Note: `--reload` makes the backend hot-reload on code edits; without it a running
process serves a stale snapshot (this is why an old answer like "1.33%" can persist
until you restart).
