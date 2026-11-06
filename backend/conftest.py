"""pytest configuration + hermetic fixtures.

Having this file at backend/ makes pytest insert backend/ on sys.path, so tests
can `import rag...` and `import app`.

Hermetic by default
-------------------
The real embedding model / tokenizer live on HuggingFace and the real index needs a
full ingest (OCR!). Unit tests must not depend on either, so by default every test
runs with:
  * a whitespace "tokenizer" in place of the bge tokenizer (rag.chunk), and
  * a deterministic hashed bag-of-words embedder in place of bge-small (rag.embed).
Tests that genuinely need the real model + a real index opt out with
`pytestmark = pytest.mark.real_index` (and are skipped if no index is built).

`tiny_index` builds a small synthetic two-document index in a temp dir and points
the store at it — no network, no PDFs, no API keys.
"""
from __future__ import annotations

import dataclasses
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.resolve()))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_index: needs the real embedding model and a built index "
        "(python ingest.py); skipped otherwise",
    )


def pytest_collection_modifyitems(config, items):
    """`real_index` tests need the real embedding stack too, not just an index on
    disk (e.g. a light CI-style install has an index but no sentence-transformers)."""
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is not None:
        return
    skip = pytest.mark.skip(reason="sentence-transformers not installed (light install)")
    for item in items:
        if item.get_closest_marker("real_index"):
            item.add_marker(skip)


class FakeTokenizer:
    """Whitespace tokenizer exposing the one method rag.chunk uses."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (L2-normalised, 128-d)."""

    DIM = 128

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture(autouse=True)
def _hermetic_models(request, monkeypatch):
    if request.node.get_closest_marker("real_index"):
        yield
        return
    import rag.chunk as chunk
    import rag.embed as embed

    monkeypatch.setattr(chunk, "_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(embed, "_model", lambda: FakeEmbedder())
    yield


def synthetic_pages():
    from rag.extract import Page

    return [
        Page(
            doc="alpha.pdf", page=1, source="text-layer",
            statement_type="standalone", fiscal_period="March 31, 2026",
            doc_label="Standalone Results — March 31, 2026",
            text=(
                "Quarter ended Year ended\nParticulars 31.03.2026 31.03.2025\n"
                "Gross NPAs to Gross Advances 1.15% 1.24% 1.33%\n"
                "Capital Adequacy Ratio 19.71% 19.55% 19.40%"
            ),
        ),
        Page(
            doc="alpha.pdf", page=2, source="ocr",
            statement_type="standalone", fiscal_period="March 31, 2026",
            doc_label="Standalone Results — March 31, 2026",
            text="The Dubai branch received a decision notice from the regulator "
                 "restricting onboarding of new customers.",
        ),
        Page(
            doc="beta.pdf", page=1, source="text-layer",
            statement_type="consolidated", fiscal_period="September 30, 2025",
            doc_label="Consolidated Results — September 30, 2025",
            text=(
                "Quarter ended Half year ended\nParticulars 30.09.2025 30.09.2024\n"
                "Gross NPAs to Gross Advances 1.24% 1.36% 1.40%\n"
                "Net profit for the period 18641.28 16820.10 15000.00"
            ),
        ),
    ]


@pytest.fixture
def tiny_index(tmp_path, monkeypatch):
    """Build a small synthetic index in tmp_path and point the store at it."""
    import rag.store as store
    from rag.chunk import chunk_pages

    cfg = dataclasses.replace(store.CONFIG, index_dir=tmp_path / "index")
    monkeypatch.setattr(store, "CONFIG", cfg)
    store.reload_index()
    chunks = chunk_pages(synthetic_pages())
    store.index_chunks(chunks)
    yield cfg.index_dir
    store.reload_index()


@pytest.fixture
def no_index(tmp_path, monkeypatch):
    """Point the store at an empty directory (no index built)."""
    import rag.store as store

    cfg = dataclasses.replace(store.CONFIG, index_dir=tmp_path / "missing")
    monkeypatch.setattr(store, "CONFIG", cfg)
    store.reload_index()
    yield cfg.index_dir
    store.reload_index()
