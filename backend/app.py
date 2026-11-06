"""FastAPI backend — the HTTP API the frontend calls.

Endpoints:
    GET  /health   -> {status, indexed_chunks, model}
    POST /ask      -> {answer, sources[], refused}

Run (from backend/):
    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.config import CONFIG
from rag.generate import LLMConfigError, LLMError, LLMRateLimitError
from rag.pipeline import AskResult, ask
from rag.ratelimit import SlidingWindowLimiter
from rag.store import IndexNotBuiltError, collection_size, list_documents

app = FastAPI(title="SourceMind", version="1.0")

# The Vite dev server runs on 5173; allow it (and localhost) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Per-client throttle for /ask (see rag/ratelimit.py). Module-level so tests can swap it.
limiter = SlidingWindowLimiter(limit=CONFIG.rate_limit_per_minute, window_seconds=60)


def _rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.check(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait {retry_after}s and try again.",
            headers={"Retry-After": str(retry_after)},
        )


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


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    refused: bool


def _to_response(r: AskResult) -> AskResponse:
    return AskResponse(
        answer=r.answer,
        refused=r.refused,
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
    return {
        # "index_missing" tells operators/UI that /ask can't work yet.
        "status": "ok" if chunks > 0 else "index_missing",
        "indexed_chunks": chunks,
        "model": CONFIG.llm_model,
        "embedding_model": CONFIG.embedding_model,
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


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(_rate_limit)])
def ask_endpoint(req: AskRequest):
    if req.doc is not None:
        known = {d["doc"] for d in list_documents()}
        if known and req.doc not in known:  # (an unbuilt index is reported below)
            raise HTTPException(status_code=404, detail=f"Unknown document: {req.doc}")
    try:
        result = ask(req.question, doc=req.doc)
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
    except LLMConfigError as e:  # server not set up for LLM calls (no API key)
        raise HTTPException(status_code=503, detail=str(e))
    except LLMError as e:  # message is curated + safe; raw provider text is only logged
        raise HTTPException(status_code=502, detail=str(e))
    return _to_response(result)
