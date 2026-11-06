"""Index persistence: atomic writes and detection of a mismatched/corrupt index."""
import json

import numpy as np
import pytest

import rag.store as store
from rag.chunk import chunk_pages
from conftest import synthetic_pages


def test_index_roundtrip_leaves_no_temp_files(tiny_index):
    names = sorted(p.name for p in tiny_index.iterdir())
    assert names == ["chunks.json", "embeddings.npy"]
    assert store.collection_size() == len(json.loads((tiny_index / "chunks.json").read_text("utf-8")))


def test_failed_write_keeps_the_previous_index(tiny_index, monkeypatch):
    before = (tiny_index / "chunks.json").read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(store.json, "dump", boom)
    with pytest.raises(OSError):
        store.index_chunks(chunk_pages(synthetic_pages()))
    assert (tiny_index / "chunks.json").read_bytes() == before
    assert sorted(p.name for p in tiny_index.iterdir()) == ["chunks.json", "embeddings.npy"]


def test_mismatched_embeddings_and_metadata_is_a_clear_error(tiny_index):
    emb = np.load(tiny_index / "embeddings.npy")
    np.save(tiny_index / "embeddings.npy", emb[:-1])  # simulate an interrupted ingest
    store.reload_index()
    with pytest.raises(store.IndexCorruptError, match="ingest.py"):
        store.search("gross npa")
    assert store.collection_size() == 0


def test_empty_chunk_list_is_rejected(tiny_index):
    with pytest.raises(ValueError, match="no chunks"):
        store.index_chunks([])
