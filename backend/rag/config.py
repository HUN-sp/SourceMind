"""Central configuration, loaded from environment (.env).

Every tunable lives here so the rest of the code never reads os.environ directly.
This is also the single place a reviewer looks to understand the knobs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above backend/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path(env_value: str) -> Path:
    """Resolve a path from .env relative to the project root if it's relative."""
    p = Path(env_value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass(frozen=True)
class Config:
    # LLM
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "8"))
    min_similarity: float = float(os.getenv("MIN_SIMILARITY", "0.30"))

    # Hybrid retrieval (Phase 5 stretch): fuse a BM25 keyword ranking with the
    # vector ranking via Reciprocal Rank Fusion. Fixes the case where a terse
    # numeric row (e.g. "Capital Adequacy Ratio 19.88%") is out-ranked by chatty
    # boilerplate that merely echoes the question's date/entity words.
    use_hybrid: bool = os.getenv("USE_HYBRID", "true").strip().lower() in ("1", "true", "yes")
    # RRF dampening constant. Lower => more weight to each list's very top ranks.
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    # When a question names NO period, gently prefer the most recent filing so a
    # generic "what was total income?" surfaces the latest quarter, not an older
    # one that happened to extract more cleanly. Bounded tiebreak (added to the
    # fused score); an explicit period in the question disables it. 0 = off.
    recency_boost: float = float(os.getenv("RECENCY_BOOST", "0.01"))

    # Cross-encoder re-ranking: after hybrid retrieval gathers a candidate pool,
    # a small model re-reads each (question, chunk) pair JOINTLY and re-orders
    # them, then we keep top_k. Catches the right numeric row when it lands just
    # outside the top-k of the first-stage ranking. ~80MB model, CPU-friendly.
    # Dropped to OFF by default: the cross-encoder reordered terse numeric value
    # rows below fluent prose decoys (e.g. an OFS "net gain" note out-ranking the
    # real "Net Profit for the period" row), hurting the highest-value questions.
    # First-stage hybrid (vector + BM25 + RRF) ranked those value rows better.
    # Re-enable with USE_RERANKER=true to A/B.
    use_reranker: bool = os.getenv("USE_RERANKER", "false").strip().lower() in ("1", "true", "yes")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Size of the first-stage candidate pool handed to the reranker.
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "25"))
    # We don't let the cross-encoder fully OVERRIDE first-stage ranking — left alone
    # it reorders by "reads like an answer", which buries terse numeric value rows
    # under fluent prose decoys (e.g. an OFS "net gain" note out-ranking the actual
    # "Net Profit for the period" row). Instead we RANK-FUSE first-stage + reranker
    # (+ a numeric-density signal for figure questions). Lower k => sharper top.
    rerank_rrf_k: int = int(os.getenv("RERANK_RRF_K", "10"))

    # Paths
    data_dir: Path = _path(os.getenv("DATA_DIR", "./data"))
    index_dir: Path = _path(os.getenv("INDEX_DIR", "./storage/index"))

    # Chunking (see chunk.py for rationale)
    chunk_max_tokens: int = 350      # target chunk size, well under bge-small's 512 window
    chunk_overlap_tokens: int = 60   # overlap so a fact split across a boundary survives

    # Ingestion: a page whose extracted text is shorter than this (chars) is
    # treated as "no usable text layer" and sent to the OCR fallback.
    ocr_char_threshold: int = 50


CONFIG = Config()
