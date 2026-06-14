"""FastAPI backend — the HTTP API the frontend calls.

Endpoints:
    GET  /health   -> {status, indexed_chunks, model}
    POST /ask      -> {answer, sources[], refused}

Run (from backend/):
    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.config import CONFIG
from rag.generate import LLMError
from rag.pipeline import AskResult, ask
from rag.store import collection_size

app = FastAPI(title="Ask-the-Docs", version="1.0")

# The Vite dev server runs on 5173; allow it (and localhost) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    doc: str | None = Field(None, description="Optional: restrict to one document")


class SourceOut(BaseModel):
    doc: str
    page: int
    snippet: str
    similarity: float
    doc_label: str = ""
    statement_type: str = ""


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
            )
            for s in r.sources
        ],
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "indexed_chunks": collection_size(),
        "model": CONFIG.llm_model,
        "embedding_model": CONFIG.embedding_model,
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    from fastapi import HTTPException

    try:
        result = ask(req.question, doc=req.doc)
    except LLMError as e:
        # The model is unreachable (bad key / rate limit). Tell the client clearly.
        raise HTTPException(status_code=502, detail=str(e))
    return _to_response(result)
