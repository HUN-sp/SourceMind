# SourceMind — a grounded RAG Q&A assistant

Ask natural-language questions about a set of PDF documents and get answers that
are **grounded in the documents**, **cite the exact source (document + page)**,
and **honestly refuse** when the answer isn't in the corpus.

The provided corpus is four HDFC Bank quarterly/annual financial filings, but the
pipeline is general — it never hardcodes anything about a specific file.

---

## Architecture at a glance

```
INGEST (offline, run once)                  ASK (per question, live)
─────────────────────────                   ─────────────────────────
data/*.pdf                                   POST /ask {question}
  │ extract text per page (pdfplumber)         │ embed question (bge-small)
  │   └─ OCR fallback if no text layer (RapidOCR)
  │ clean + de-glue OCR words                  │ hybrid search:
  │ derive metadata (statement type,           │   vector (cosine) + BM25 keyword
  │   period, doc label) from headers          │   fused by Reciprocal Rank Fusion
  │ chunk per page (≤350 tok, header-attached) │   + recency tiebreak
  │ embed (local bge-small) + store            │
  ▼                                            │ GATE 1: best similarity < floor → refuse
storage/index/                                 │ build numbered, labelled context
  ├─ embeddings.npy  (N×384, normalized)       │ LLM answers ONLY from passages (Groq)
  └─ chunks.json     (text + metadata)         │ GATE 2: LLM says not_covered → refuse
                                               ▼
                                       { answer, sources[], refused }
```

The two halves share **one** thing — the embedding model — which is the only
reason similarity search is meaningful. Full rationale for every choice is in
[`DECISIONS.md`](DECISIONS.md).

| Layer | Choice |
|---|---|
| PDF parsing | `pdfplumber` (layout-preserving) + RapidOCR fallback for scans |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free, no key) |
| Vector store | exact cosine over a NumPy matrix (no native deps) |
| Keyword search | `rank-bm25`, fused with vectors via Reciprocal Rank Fusion |
| LLM | Groq free tier (`llama-3.3-70b-versatile`), temperature 0, JSON mode |
| Backend | FastAPI (`/health`, `/ask`) |
| Frontend | React + Vite |

---

## Setup from zero

### 0. Prerequisites
- Python 3.10+ and Node 18+
- A free Groq API key — sign up at <https://console.groq.com/keys> (no card needed)

### 1. Backend
```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate    # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# configure secrets
cp ../.env.example ../.env          # then edit ../.env and paste your GROQ_API_KEY
```

### 2. Ingest the documents (one-shot, builds the search index)
```bash
# from backend/
python ingest.py
```
This parses every PDF in `../data`, OCRs scanned pages, chunks, embeds, and writes
the index to `../storage/index`. The first run downloads the embedding model
(~130 MB) and OCRs the scanned PDF, so it takes a few minutes. Re-run it whenever
the documents or chunking change.

```bash
python ingest.py --stats     # print page/chunk stats without indexing
```

### 3. Run the backend
```bash
# from backend/
uvicorn app:app --reload --port 8000
```
- `GET  http://localhost:8000/health` → index + model status
- `POST http://localhost:8000/ask` with `{"question": "..."}` → `{answer, sources[], refused}`

### 4. Run the frontend
```bash
cd frontend
npm install
npm run dev      # opens http://localhost:5173
```
Vite proxies `/api/*` to the backend on port 8000, so the browser sees one origin.

---

## Inspecting and testing (no Groq tokens needed)

```bash
cd backend

# See exactly what retrieval returns for a question (no LLM call):
python query.py "What is the gross NPA ratio?" --k 8

# Token-free retrieval harness: does the answer chunk land in the top-8 for the
# 20-question evaluation bank? Great fast feedback loop while tuning.
python retrieval_harness.py
python retrieval_harness.py --rerank     # A/B with the cross-encoder reranker on

# Automated tests (chunking/provenance, retrieval, API/grounding):
pytest
```

End-to-end question runs that DO call the LLM:
```bash
python run_batch.py      # runs the evaluation question bank against the live API
```
> Note: the Groq free tier has a ~100k-tokens/day cap. Each answer sends several
> document chunks as context, so a long testing session can exhaust the daily
> budget — at which point the API returns a clean "model unavailable" 502 (not a
> crash). Use a fresh key or wait for the daily reset to continue.

---

## Configuration

All tunables live in `backend/rag/config.py`, overridable via `.env` (see
[`.env.example`](.env.example)). Key knobs: `TOP_K`, `MIN_SIMILARITY` (grounding
floor), `USE_HYBRID`, `RRF_K`, `RECENCY_BOOST`, `USE_RERANKER`.

## Repo layout
```
data/                 the PDF corpus
storage/index/        built by ingest.py (embeddings.npy + chunks.json)
backend/
  app.py              FastAPI server
  ingest.py           PDF -> index CLI
  query.py            retrieval inspector
  retrieval_harness.py token-free top-k evaluation
  rag/                config, extract, chunk, embed, store, generate, pipeline
  tests/              pytest suite
frontend/             React + Vite UI
docs/                 ARCHITECTURE.md, JOURNEY.md, STUDY_GUIDE.md
DECISIONS.md          why each choice was made (read this)
EVALUATION.md         8+ questions with answers, sources, honest verdicts
```
