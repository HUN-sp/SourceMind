# SourceMind — progress log

Working branch: `claude/sourcemind-improvements` (PR target: `main`).
This file is the resume point: re-read it and continue from the first unchecked item.

## 1. What the project is

SourceMind is a grounded RAG question-answering app over PDF documents (the bundled
corpus is four HDFC Bank quarterly/annual results filings in `data/`). It returns an
answer with `(document, page)` citations, and refuses when the answer isn't in the
documents.

- **Ingest** (`backend/ingest.py`): pdfplumber text per page → RapidOCR fallback for
  scanned pages → clean/de-glue → derive statement type + period → chunk (≤350 tokens,
  page-bounded, period header re-attached) → embed locally (`BAAI/bge-small-en-v1.5`) →
  `storage/index/{embeddings.npy,chunks.json}`.
- **Ask** (`backend/rag/pipeline.py`): hybrid retrieval (cosine + BM25 fused with RRF,
  recency tiebreak) → gate 1 (similarity floor) → Groq LLM (`llama-3.3-70b-versatile`,
  JSON mode) → gate 2 (`not_covered`) → citations.
- **API** (`backend/app.py`, FastAPI): `GET /health`, `POST /ask`.
- **UI** (`frontend/`, React + Vite): ask box, answer panel, source cards. Vite proxies
  `/api/*` → `localhost:8000`.

## 2. How to run / test

```bash
# backend (Python 3.10–3.12; rapidocr-onnxruntime has no wheels for 3.13)
cd backend && python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env          # add a GROQ_API_KEY (free tier)
python ingest.py                    # builds ../storage/index (downloads ~130MB model, OCRs data_4)
uvicorn app:app --reload --port 8000
# frontend
cd frontend && npm install && npm run dev      # http://localhost:5173
# tests
cd backend && pytest
```

## 3. Environment notes for this session

- `huggingface.co` is blocked by the sandbox egress proxy, so the embedding model and
  tokenizer cannot be downloaded → **the real index cannot be built here** and the
  real-model tests/harness cannot be run. See "Needs real testing".
- Python 3.12 venv used (3.13 is the system default and can't install RapidOCR).

## 4. Problems found (ranked) — Phase 1

Legend: `[ ]` open, `[x]` fixed. Severity: **H**igh / **M**edium / **L**ow.

### High
- [x] **H1** `backend/retrieval_harness.py:1-184` — a PowerShell terminal transcript was
  pasted over the top of the file; the script has a `SyntaxError` at line 1 and cannot run
  (README documents it as a main dev tool).
- [x] **H2** `backend/query.py:43` — `rerank=not args.no_rerank` forces the cross-encoder
  **on** by default (and downloads an 80 MB model), contradicting `config.py:60`
  (default off) and DECISIONS §6. The inspector therefore shows different results from
  the API.
- [x] **H3** `.env.example:38` — ships `USE_RERANKER=true`, whereas the code default and
  DECISIONS §6 say it must be off (it measurably hurts numeric value rows). Anyone who
  follows the README setup gets the worse behaviour.
- [ ] **H4** `backend/rag/store.py:87` + `backend/app.py:81-86` — if the index isn't
  built, `/ask` raises an unhandled `FileNotFoundError` → bare HTTP 500; `/health`
  (`app.py:69-74`) still reports `"status": "ok"`.
- [ ] **H5** `backend/rag/pipeline.py:126-133,176,185` — `parse_llm_json` accepts any JSON
  value; if the model returns a list/string, or `used_passages` is a non-list (e.g. `3`),
  `data.get` / iteration raises → HTTP 500 instead of the documented fail-safe refusal.
- [ ] **H6** `backend/app.py:33` — no max length on `question` (unbounded text goes to the
  embedder and, via the prompt, to the paid LLM) and no rate limit on `/ask` (anyone who
  can reach the port can burn the Groq quota). Security / cost risk.

### Medium
- [ ] **M1** `backend/rag/store.py:71-73` — index written non-atomically (`.npy` then
  `.json`); an interrupted ingest leaves a mismatched pair that loads silently
  (`store.py:90-93` never checks `len(emb) == len(meta)`).
- [ ] **M2** `backend/ingest.py:46` — `min(tok_counts)` raises `ValueError` when there are
  zero chunks (all pages empty); `backend/rag/extract.py:210-211` — one unreadable PDF
  aborts the whole ingest with a raw traceback.
- [ ] **M3** `backend/rag/chunk.py:150` — oversized-block splitter can append an empty
  chunk; `chunk.py:146-154,121-125` re-tokenize the growing string on every word (O(n²)).
- [ ] **M4** `backend/rag/extract.py:102-109` — the PDF is re-opened for every OCR page and
  not closed if rasterisation/OCR raises (leaks handles; slow for big scans).
- [ ] **M5** `backend/app.py:85` — raw provider exception text is returned to the client
  (info leak) and every LLM failure is a 502, including rate limits that deserve a 429
  with a clear message.
- [ ] **M6** `backend/requirements.txt` — `rapidocr-onnxruntime==1.4.4` cannot install on
  Python 3.13 (README says "3.10+"); pins are ~1 year old (fastapi 0.115.6, groq 0.13.1,
  sentence-transformers 3.3.1); `pytest`/`httpx` mixed into runtime deps.
- [ ] **M7** Tests are not hermetic and thin: `backend/conftest.py:12` forces HF offline so
  `tests/test_chunking.py` fails on a machine without a cached tokenizer;
  `test_retrieval.py`/`test_pipeline.py` skip when no index exists (so CI would test
  nothing); no tests for `/ask`, `parse_llm_json`, `_query_tokens`, `_rrf_fuse`,
  `_period_key`, recency, the `doc` filter's error path, or index persistence. No CI.

### Low
- [ ] **L1** `backend/ingest.py:29` `__import__("pathlib")` hack; `backend/app.py:79`
  import inside the handler.
- [ ] **L2** `backend/rag/generate.py:44` passes `response_format=None` when
  `json_mode=False` (SDK expects it omitted) — latent bug, parameter currently unused.
- [ ] **L3** `backend/rag/config.py:30-62` — a malformed numeric env var (`TOP_K=abc`)
  crashes at import with an unhelpful `ValueError`.
- [ ] **L4** `frontend/src/api.js:7` — no timeout/abort; a hung backend leaves the UI
  spinning forever.
- [ ] **L5** `frontend/src/App.jsx:38` — handler named `useExample` (looks like a hook);
  `Enter` submits during IME composition (`App.jsx:59-64`).
- [ ] **L6** `frontend/package.json:2` — stale package name `ask-the-docs-frontend`.
- [ ] **L7** `backend/rag/store.py:296` — doc filter builds a Python list over all chunks
  on every query; `Source.fiscal_period` is computed but never exposed by the API.
- [ ] **L8** `backend/rag/extract.py:50-57` — generic garbage-line filter may drop short
  genuine rows that are mostly punctuation (e.g. `( - )`); documented trade-off, left as is.
- [ ] **L9** `Video_Link` — extension-less file at repo root holding a URL (cosmetic; left).

## 5. Fix log — Phase 2
(commits are listed here as items are completed)
- H1 fixed: removed pasted PowerShell transcript from `retrieval_harness.py`; added `tests/test_scripts_compile.py` (compiles every backend `.py`). Harness itself not run (needs the real index).
- H2 fixed: `query.py` follows `CONFIG.use_reranker` unless `--rerank`/`--no-rerank` is given (`tests/test_query_cli.py`).
- H3 fixed: `.env.example` now ships `USE_RERANKER=false`; `tests/test_env_example.py` keeps `.env.example` and code defaults in sync.

## 6. Enhancements — Phase 3
(chosen after Phase 2)

## 7. Needs real testing
- Building the real index (`python ingest.py`) and anything depending on it:
  `tests/test_retrieval.py`, `tests/test_pipeline.py`, `retrieval_harness.py`, `query.py`
  — needs `huggingface.co` access for `BAAI/bge-small-en-v1.5` (blocked in this sandbox).
- Any real Groq call (`/ask` end-to-end, `run_batch.py`) — needs a real `GROQ_API_KEY`.
- OCR of the scanned PDFs (`data_4.pdf`) via RapidOCR.
