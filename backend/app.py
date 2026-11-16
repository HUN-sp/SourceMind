"""FastAPI backend — the HTTP API the frontend calls.

Endpoints:
    GET  /health   -> {status, indexed_chunks, model}
    POST /ask      -> {answer, sources[], refused}

Run (from backend/):
    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import demo, store
from rag.cache import AnswerCache
from rag.config import CONFIG
from rag.generate import LLMConfigError, LLMError, LLMModelError, LLMRateLimitError, llm_configured
from rag.generate import generate as llm_generate
from rag.pipeline import AskResult, ask
from rag.quota import LLMQuota, QuotaExceeded
from rag.ratelimit import SlidingWindowLimiter
from rag.store import IndexNotBuiltError, collection_size, list_documents

def _warm_up() -> None:
    """Load the index, BM25 and the embedding model so the first visitor isn't
    the one paying the ~10 s cold start (on a small server)."""
    try:
        store.search("warm up")
    except Exception as e:  # never block startup; /health reports real problems
        logging.getLogger(__name__).warning("warm-up skipped: %s", e)


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("WARMUP", "false").strip().lower() in ("1", "true", "yes"):
        threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="SourceMind", version="1.0", lifespan=lifespan)

# The Vite dev server runs on 5173; allow it (and localhost) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Per-client throttle for /ask (see rag/ratelimit.py). Module-level so tests can swap it.
limiter = SlidingWindowLimiter(limit=CONFIG.rate_limit_per_minute, window_seconds=60)


def _client_ip(request: Request) -> str:
    # Behind Caddy, uvicorn runs with --proxy-headers so this is the visitor's real IP.
    return request.client.host if request.client else "unknown"


# Quota for LIVE LLM calls only (see rag/quota.py). Module-level so tests can swap it.
llm_quota = LLMQuota(CONFIG.llm_limit_per_ip, CONFIG.llm_limit_global)

# Pre-computed responses for the demo chips (see rag/demo.py); loaded lazily so a
# freshly built index is picked up. None = not loaded yet.
_preset_cache: dict[str, dict] | None = None


def preset_responses() -> dict[str, dict]:
    global _preset_cache
    if _preset_cache is None:
        _preset_cache = demo.load_cache()
    return _preset_cache


def _rate_limit(request: Request) -> None:
    client_ip = _client_ip(request)
    allowed, retry_after = limiter.check(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait {retry_after}s and try again.",
            headers={"Retry-After": str(retry_after)},
        )


# Remembers answered questions (see rag/cache.py). Module-level so tests can swap it.
answer_cache = AnswerCache(CONFIG.cache_size)


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=CONFIG.max_question_chars,
        description="The user's question",
    )
    doc: str | None = Field(
        None, max_length=255, description="Optional: restrict to one document"
    )


class SourceOut(BaseModel):
    doc: str
    page: int
    snippet: str
    similarity: float
    doc_label: str = ""
    statement_type: str = ""
    fiscal_period: str = ""


class RetrievedOut(BaseModel):
    """One retrieved chunk with the score each first-stage signal gave it."""
    doc: str
    page: int
    doc_label: str = ""
    chunk_type: str = ""
    snippet: str
    cosine: float                 # BGE-small cosine similarity
    vec_rank: int | None = None
    bm25: float | None = None
    bm25_rank: int | None = None
    rrf: float | None = None      # fused Reciprocal Rank Fusion score (final order)
    cited: bool = False           # did the answer cite this chunk?


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    refused: bool
    cached: bool = False  # True when served from the answer cache (no LLM tokens spent)
    preset: str | None = None     # id of the demo preset this response came from
    retrieval: list[RetrievedOut] = []   # what retrieval surfaced, in final rank order
    top_similarity: float = 0.0          # best cosine (what grounding gate #1 checks)
    min_similarity: float = CONFIG.min_similarity
    latency_ms: int | None = None        # server time to produce the response


_SNIPPET_CHARS = 600


def _to_response(r: AskResult) -> AskResponse:
    cited = {(s.doc, s.page, s.snippet) for s in r.sources}
    return AskResponse(
        answer=r.answer,
        refused=r.refused,
        top_similarity=round(r.top_similarity, 3),
        retrieval=[
            RetrievedOut(
                doc=c.doc, page=c.page, doc_label=c.doc_label, chunk_type=c.chunk_type,
                snippet=c.text if len(c.text) <= _SNIPPET_CHARS else c.text[:_SNIPPET_CHARS] + "…",
                cosine=round(c.similarity, 4), vec_rank=c.vec_rank,
                bm25=None if c.bm25 is None else round(c.bm25, 3), bm25_rank=c.bm25_rank,
                rrf=None if c.rrf is None else round(c.rrf, 5),
                cited=(c.doc, c.page, c.text) in cited,
            )
            for c in r.retrieved
        ],
        sources=[
            SourceOut(
                doc=s.doc, page=s.page, snippet=s.snippet,
                similarity=round(s.similarity, 3),
                doc_label=s.doc_label, statement_type=s.statement_type,
                fiscal_period=s.fiscal_period,
            )
            for s in r.sources
        ],
    )


@app.get("/health")
def health() -> dict:
    chunks = collection_size()
    has_key = llm_configured()
    problems = []
    if chunks == 0:
        problems.append("The search index isn't built. Run `python ingest.py` in backend/.")
    if not has_key:
        problems.append(
            "No GROQ_API_KEY is configured. Copy .env.example to .env, add a free key "
            "from https://console.groq.com/keys, and restart the server."
        )
    return {
        # "index_missing" tells operators/UI that /ask can't work yet.
        "status": "ok" if chunks > 0 else "index_missing",
        "ready": not problems,
        "problems": problems,
        "llm_configured": has_key,
        "indexed_chunks": chunks,
        "model": CONFIG.llm_model,
        "embedding_model": CONFIG.embedding_model,
        "cache": answer_cache.stats(),
    }


class DocumentOut(BaseModel):
    doc: str
    doc_label: str = ""
    fiscal_period: str = ""
    pages: int
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentOut]


@app.get("/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    """The indexed documents, so a client can offer a 'search only in…' picker."""
    return DocumentsResponse(documents=[DocumentOut(**d) for d in list_documents()])


@app.get("/demo")
def demo_info() -> dict:
    """The one-click demo presets (and whether each is served instantly from the
    pre-computed file) plus the evaluation summary for the inspector panel."""
    cached = preset_responses()
    return {
        "presets": [{**p, "instant": p["id"] in cached} for p in demo.PRESETS],
        "evals": demo.load_evals(),
        "limits": {"llm_per_ip": CONFIG.llm_limit_per_ip, "llm_global": CONFIG.llm_limit_global},
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(_rate_limit)])
def ask_endpoint(req: AskRequest, request: Request):
    started = time.perf_counter()

    def _ms() -> int:
        return round((time.perf_counter() - started) * 1000)

    preset = demo.preset_for(req.question, req.doc)
    if preset is not None:
        stored = preset_responses().get(preset["id"])
        if stored is not None:  # pre-computed by the real pipeline; zero LLM tokens
            return AskResponse(**{**stored, "cached": True, "preset": preset["id"], "latency_ms": _ms()})

    if req.doc is not None:
        known = {d["doc"] for d in list_documents()}
        if known and req.doc not in known:  # (an unbuilt index is reported below)
            raise HTTPException(status_code=404, detail=f"Unknown document: {req.doc}")
    key = AnswerCache.key(req.question, req.doc, store.generation())
    hit = answer_cache.get(key)
    if hit is not None:
        return hit.model_copy(update={"cached": True, "latency_ms": _ms()})

    client_ip = _client_ip(request)

    def metered_generate(system: str, user: str) -> str:
        # Charged only when the pipeline actually calls the LLM (gate #1 refusals are free).
        llm_quota.charge(client_ip)
        return llm_generate(system, user)

    try:
        result = ask(req.question, doc=req.doc, generate=metered_generate)
    except QuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e), headers={"Retry-After": str(e.retry_after)})
    except IndexNotBuiltError:
        # Fixed text on purpose: the exception message contains a server path.
        raise HTTPException(
            status_code=503,
            detail="The search index is missing or unreadable. Run `python ingest.py` "
                   "on the server, then try again.",
        )
    except LLMRateLimitError as e:
        headers = {"Retry-After": str(e.retry_after)} if e.retry_after else None
        raise HTTPException(status_code=429, detail=str(e), headers=headers)
    except (LLMConfigError, LLMModelError) as e:  # server not set up for LLM calls (no key / bad model)
        raise HTTPException(status_code=503, detail=str(e))
    except LLMError as e:  # message is curated + safe; raw provider text is only logged
        raise HTTPException(status_code=502, detail=str(e))
    response = _to_response(result)
    if preset is not None:
        response.preset = preset["id"]
    response.latency_ms = _ms()
    if not response.refused:  # never cache refusals: they may be a transient LLM hiccup
        answer_cache.put(key, response)
    return response
