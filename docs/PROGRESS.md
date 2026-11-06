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
- [x] **H4** `backend/rag/store.py:87` + `backend/app.py:81-86` — if the index isn't
  built, `/ask` raises an unhandled `FileNotFoundError` → bare HTTP 500; `/health`
  (`app.py:69-74`) still reports `"status": "ok"`.
- [x] **H5** `backend/rag/pipeline.py:126-133,176,185` — `parse_llm_json` accepts any JSON
  value; if the model returns a list/string, or `used_passages` is a non-list (e.g. `3`),
  `data.get` / iteration raises → HTTP 500 instead of the documented fail-safe refusal.
- [x] **H6** `backend/app.py:33` — no max length on `question` (unbounded text goes to the
  embedder and, via the prompt, to the paid LLM) and no rate limit on `/ask` (anyone who
  can reach the port can burn the Groq quota). Security / cost risk.

### Medium
- [x] **M1** `backend/rag/store.py:71-73` — index written non-atomically (`.npy` then
  `.json`); an interrupted ingest leaves a mismatched pair that loads silently
  (`store.py:90-93` never checks `len(emb) == len(meta)`).
- [x] **M2** `backend/ingest.py:46` — `min(tok_counts)` raises `ValueError` when there are
  zero chunks (all pages empty); `backend/rag/extract.py:210-211` — one unreadable PDF
  aborts the whole ingest with a raw traceback.
- [x] **M3** `backend/rag/chunk.py:150` — oversized-block splitter can append an empty
  chunk; `chunk.py:146-154,121-125` re-tokenize the growing string on every word (O(n²)).
- [x] **M4** `backend/rag/extract.py:102-109` — the PDF is re-opened for every OCR page and
  not closed if rasterisation/OCR raises (leaks handles; slow for big scans).
- [x] **M5** `backend/app.py:85` — raw provider exception text is returned to the client
  (info leak) and every LLM failure is a 502, including rate limits that deserve a 429
  with a clear message.
- [x] **M6** (partly — see note) `backend/requirements.txt` — `rapidocr-onnxruntime==1.4.4` cannot install on
  Python 3.13 (README says "3.10+"); pins are ~1 year old (fastapi 0.115.6, groq 0.13.1,
  sentence-transformers 3.3.1); `pytest`/`httpx` mixed into runtime deps.
- [x] **M7** (CI workflow is tracked as enhancement E5) Tests are not hermetic and thin: `backend/conftest.py:12` forces HF offline so
  `tests/test_chunking.py` fails on a machine without a cached tokenizer;
  `test_retrieval.py`/`test_pipeline.py` skip when no index exists (so CI would test
  nothing); no tests for `/ask`, `parse_llm_json`, `_query_tokens`, `_rrf_fuse`,
  `_period_key`, recency, the `doc` filter's error path, or index persistence. No CI.

### Low
- [x] **L1** `backend/ingest.py:29` `__import__("pathlib")` hack; `backend/app.py:79`
  import inside the handler.
- [x] **L2** `backend/rag/generate.py:44` passes `response_format=None` when
  `json_mode=False` (SDK expects it omitted) — latent bug, parameter currently unused.
- [x] **L3** `backend/rag/config.py:30-62` — a malformed numeric env var (`TOP_K=abc`)
  crashes at import with an unhelpful `ValueError`.
- [x] **L4** `frontend/src/api.js:7` — no timeout/abort; a hung backend leaves the UI
  spinning forever.
- [x] **L5** `frontend/src/App.jsx:38` — handler named `useExample` (looks like a hook);
  `Enter` submits during IME composition (`App.jsx:59-64`).
- [x] **L6** `frontend/package.json:2` — stale package name `ask-the-docs-frontend`.
- [x] **L7** (fiscal_period exposed; per-query doc-filter list left — negligible at ~200 chunks) `backend/rag/store.py:296` — doc filter builds a Python list over all chunks
  on every query; `Source.fiscal_period` is computed but never exposed by the API.
- [ ] **L8** `backend/rag/extract.py:50-57` — generic garbage-line filter may drop short
  genuine rows that are mostly punctuation (e.g. `( - )`); documented trade-off, left as is.
- [ ] **L9** `Video_Link` — extension-less file at repo root holding a URL (cosmetic; left).

## 5. Fix log — Phase 2
(commits are listed here as items are completed)
- H1 fixed: removed pasted PowerShell transcript from `retrieval_harness.py`; added `tests/test_scripts_compile.py` (compiles every backend `.py`). Harness itself not run (needs the real index).
- H2 fixed: `query.py` follows `CONFIG.use_reranker` unless `--rerank`/`--no-rerank` is given (`tests/test_query_cli.py`).
- H3 fixed: `.env.example` now ships `USE_RERANKER=false`; `tests/test_env_example.py` keeps `.env.example` and code defaults in sync.
- M7 (partial): added hermetic fixtures in `conftest.py` (fake tokenizer + hash embedder + synthetic `tiny_index`); real-model tests carry `@pytest.mark.real_index`. `tests/test_chunking.py` now passes without HuggingFace.
- H4 fixed: missing index -> `/ask` returns 503 with a clear message (new `IndexNotBuiltError`), `/health` reports `index_missing`; tests in `tests/test_api.py`.
- H5 fixed: `parse_llm_json` rejects non-object JSON; strict `not_covered` and `used_passages` handling (a `"false"` string used to cause a refusal; scalar/garbage `used_passages` used to 500). `tests/test_llm_parsing.py`.
- H6 fixed: `MAX_QUESTION_CHARS` (default 1000, 422 above) and per-IP `RATE_LIMIT_PER_MINUTE` (default 30, 429 + Retry-After) on `/ask` via new `rag/ratelimit.py`; `tests/test_limits.py`. Limiter is in-memory, per process, keyed on the socket IP (behind a reverse proxy, rate-limit at the proxy).
- M1 fixed: index written via temp files + `os.replace`; `_load()` validates embeddings/chunks agree (`IndexCorruptError` -> 503 with re-ingest hint); empty chunk list rejected. `tests/test_store_index.py`.
- M2 fixed: `extract_corpus` skips (and logs) unreadable PDFs and raises a clear `RuntimeError` if nothing is usable; `ingest.py` refactored into `run()` returning an exit code (no more `min([])` crash, no `__import__` hack -> also closes L1 ingest half). Tests: `test_extract_robustness.py`, `test_ingest_cli.py`.
- M3 fixed: oversized-block splitting uses `_split_by_tokens` (binary search; never emits an empty chunk; tokenizer calls drop from ~1 per word to O(pieces·log n)). `tests/test_chunk_edge_cases.py`.
- M4 fixed: new `_OcrDocument` opens each scanned PDF once (lazily) and closes it in a `finally`; digital PDFs never import fitz/RapidOCR. `tests/test_ocr_session.py` (fake fitz + RapidOCR — real libs not run here).
- M5 + L2 fixed: `generate.py` raises typed `LLMError` subclasses with curated messages (raw provider text only logged, API key scrubbed); `/ask` maps rate limit -> 429 (+`Retry-After`), missing key -> 503, others -> 502; `response_format` is omitted (not `None`) when `json_mode=False`. `tests/test_llm_errors.py` (uses real groq exception classes, no network).
- M6 partly fixed: split dev deps into `requirements-dev.txt`; documented Python 3.10–3.12; bumped fastapi 0.115.6→0.162.0, uvicorn 0.34→0.54, pydantic 2.10→2.14.3, python-dotenv 1.0→1.2.4, groq 0.13→1.7.0 (full test-suite green on these versions). NOT bumped (cannot be verified here, see "Needs real testing"): pdfplumber, PyMuPDF, rapidocr-onnxruntime, sentence-transformers, rank-bm25.
- M7 fixed: hermetic suite now covers retrieval helpers, hybrid fusion, recency tiebreak, doc filter, gate #1, prompt ordering (`tests/test_retrieval_unit.py`) plus all the tests above; 122 pass offline, 7 real-index tests skip.
- L1 + L3 fixed: no more `__import__`/in-handler import; malformed numeric env vars raise `ValueError: TOP_K must be a int, got 'abc'` (`tests/test_config_env.py`).
- L4 + L5 + L6 fixed (+ extra bug found): `api.js` has a 60s timeout + caller cancel (`AbortSignal`), FastAPI list-style 422 `detail` no longer renders as `[object Object]`; Enter during IME composition no longer submits; `useExample` -> `runExample`; package renamed `sourcemind-frontend`. Added Vitest + Testing Library (`npm test`, 11 tests).


## 6. Enhancements — Phase 3

Chosen for real users, in build order. `[x]` = built, tested and committed.

- [x] **E1 Document picker** — the API already accepts `doc`, but the UI can't use it; the
  corpus is four filings of the same bank, so letting users scope a question to one filing
  prevents cross-period mix-ups. (`GET /documents`, 404 for unknown docs, dropdown in UI.)
- [x] **E2 Answer cache** — the Groq free tier caps ~100k tokens/day (README's #1 stated
  limitation); repeated/example questions needlessly burn it. In-memory LRU keyed on
  (question, doc), cleared on re-index, `cached` flag in the response. (`CACHE_SIZE`.)
- [x] **E3 Readiness banner** — first-run failures (backend down, index not built, no API
  key) currently surface as confusing errors after the first question. `/health` now says
  what's wrong and the UI shows a setup banner up front.
- [x] **E4 Cancel + copy + cached badge** — LLM calls take seconds: Cancel button (uses the
  AbortSignal support), a Copy-answer button (answer + citations), and an "Instant answer"
  badge for cache hits. (Scope changed from "period tags": a source card's title already
  carries statement type + period via `doc_label`, so that would have been redundant.)
- [x] **E5 CI + docs refresh** — GitHub Actions running the offline backend tests and the
  frontend tests/build on every push/PR; README gets the API reference, test instructions
  and new config knobs.

### Enhancement log
- E5 built: `.github/workflows/ci.yml` (offline backend suite from `backend/requirements-ci.txt` + frontend `npm ci`/test/build); `tests/test_requirements.py` keeps CI pins in sync; README gained the API reference, test/CI instructions and new knobs; stale 502 wording fixed in README/DECISIONS/ARCHITECTURE/STUDY_GUIDE. Verified locally in a fresh venv (149 passed) and fresh `npm ci`; the workflow itself has NOT run on GitHub yet.
- E4 built: Cancel button during a request (no error shown on cancel), Copy-answer button (`formatForCopy`), "Instant answer" badge; `AnswerPanel.test.jsx` + cancel test in `App.test.jsx`.
- E3 built: `/health` now returns `ready`, `problems[]`, `llm_configured` (never the key); UI shows a "Setup needed" banner (backend unreachable / index missing / no API key); bare 5xx from the dev proxy now says to check the backend. Tests in `test_api.py`, `App.test.jsx`, `api.test.js`.
- E2 built: `rag/cache.py` LRU (`CACHE_SIZE`, default 128, 0 = off) in front of `/ask`; only answered results cached, keyed on normalised (question, doc) + index generation; `cached` flag in the response and cache stats in `/health`. `backend/tests/test_cache.py`.
- E1 built: `GET /documents`, 404 for an unknown `doc` (before any LLM call), "Search in" picker in the UI. Tests: `backend/tests/test_documents.py`, `App.test.jsx`, `api.test.js`.

## 7. Needs real testing

Everything below could not be verified in this sandbox (no HuggingFace access, no real
Groq key, no GPU/OCR models). Nothing here is claimed to work.

- **Real index build + retrieval quality.** `python ingest.py` with the real
  `BAAI/bge-small-en-v1.5` model, then `tests/test_retrieval.py`, `tests/test_pipeline.py`
  (tagged `real_index`), `python query.py ...` and `python retrieval_harness.py`.
  The harness file was only fixed to *compile*; its 10/17 recall figure is the
  author's, not re-measured. `EVALUATION.md` numbers are likewise not re-run.
- **Real LLM calls.** `/ask` end-to-end and `run_batch.py` need a real `GROQ_API_KEY`.
  `groq` was bumped 0.13.1 → 1.7.0; the error classes were exercised via the SDK's real
  exception types, but an actual completion call (`seed`, `response_format`) was not.
- **OCR path with the real libraries.** `_OcrDocument` (open-once/close) is tested with fake
  `fitz`/RapidOCR modules only; run `python ingest.py` on `data_4.pdf` to confirm.
- **PDF/ML pins unchanged** (`pdfplumber`, `PyMuPDF`, `rapidocr-onnxruntime`,
  `sentence-transformers`, `rank-bm25`) — still ~1 year old; bump them only alongside a
  real ingest + harness run. The full `pip install -r requirements.txt` was not completed
  here (torch download too slow); only the light CI subset was installed and tested.
- **GitHub Actions workflow** (`.github/workflows/ci.yml`) is syntactically valid YAML and
  its commands pass locally, but it has not run on GitHub yet.
- **Rate limiter behind a reverse proxy** keys on the socket IP (all users share one key
  behind a proxy) — rate-limit at the proxy in that setup.

What *was* verified end-to-end here: the real FastAPI app + store + pipeline + Vite dev
proxy + React UI driven by Playwright in Chromium, with a fake embedder and fake LLM on a
synthetic two-document index (ask, scoped ask, cancel, copy, cache badge, refusal,
backend-down banner, no JS console errors).

## 8. Decisions made while working autonomously
- **Branch:** the task named `claude/sourcemind-improvements` explicitly, so work was done
  there (not on the session's default `claude/exciting-darwin-r8yf2s`). Never pushed to `main`.
- **Hermetic tests over real-model tests:** unit tests use fake models by default; the two
  real-index files keep working unchanged via the `real_index` marker.
- **Only answered results are cached** (refusals can be transient LLM failures).
- **Rate limit defaults** (30/min/IP) and **max question length** (1000 chars) are
  conservative guesses — tune via `.env`.
- **E4 scope changed** (period tags dropped as redundant with `doc_label`).
- **L8** (garbage-line filter might drop short punctuation-heavy rows) and **L9**
  (`Video_Link` file) deliberately left alone: both are documented author trade-offs /
  cosmetic, and changing the filter would alter extraction output we can't re-verify here.
- No features were removed.

## 9. Final status
- Backend: `pytest` → 149 passed, 7 skipped (the 7 are the `real_index` tests).
- Frontend: `npm test` → 27 passed; `npm run build` OK.
