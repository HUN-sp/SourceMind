"""Chunker edge cases: no empty chunks, no lost text, and tokenization cost that
doesn't explode on one huge block."""
import dataclasses

import rag.chunk as chunk
from rag.extract import Page


def _page(text):
    return Page(doc="d.pdf", page=1, text=text, source="text-layer")


def test_single_word_longer_than_the_window_does_not_create_empty_chunks(monkeypatch):
    monkeypatch.setattr(chunk, "CONFIG", dataclasses.replace(chunk.CONFIG, chunk_max_tokens=10, chunk_overlap_tokens=0))
    monkeypatch.setattr(chunk, "count_tokens", lambda t: len(t))  # char-based "tokens"
    out = chunk.chunk_page(_page("A" * 25 + " tail words here"))
    assert out and all(c.text.strip() for c in out)
    joined = " ".join(c.text for c in out)
    assert "A" * 25 in joined and "tail words here" in joined  # nothing lost


def test_oversized_block_is_split_without_losing_words(monkeypatch):
    words = [f"w{i}" for i in range(500)]
    out = chunk.chunk_page(_page(" ".join(words)))
    assert len(out) > 1
    seen = " ".join(c.text for c in out).split()
    assert set(words) <= set(seen)


def test_oversized_block_tokenization_cost_is_not_quadratic(monkeypatch):
    calls = {"n": 0}
    real = chunk.count_tokens

    def counting(t):
        calls["n"] += 1
        return real(t)

    monkeypatch.setattr(chunk, "count_tokens", counting)
    chunk.chunk_page(_page(" ".join(f"w{i}" for i in range(3000))))
    # Per-word counting needed >3000 calls; binary search needs only O(chunks*log n).
    assert calls["n"] < 400, calls["n"]
