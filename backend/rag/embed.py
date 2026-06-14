"""Local embeddings via sentence-transformers (free, no API key).

Why bge-small-en-v1.5:
  * Small (~130MB) and CPU-friendly, so a reviewer can ingest without a GPU.
  * 512-token window — twice MiniLM's 256 — which matters because financial
    tables are dense and we want a whole statement to fit in one chunk.
  * Strong retrieval quality for its size on the MTEB benchmark.

Asymmetry: bge models are trained so that QUERIES get a short instruction prefix
while PASSAGES do not. Embedding the query with that prefix measurably improves
recall, so we expose separate document/query methods.
"""
from __future__ import annotations

from functools import lru_cache

from .config import CONFIG

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CONFIG.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts (passages). Normalized so cosine == dot product."""
    vecs = _model().encode(
        texts, normalize_embeddings=True, show_progress_bar=len(texts) > 50
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user question, with the bge query-side instruction prefix."""
    vec = _model().encode(_QUERY_PREFIX + text, normalize_embeddings=True)
    return vec.tolist()
