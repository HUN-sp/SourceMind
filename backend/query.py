"""Manual retrieval inspector — ask the index a question and see what comes back.

This is NOT the API or the LLM. It only shows the raw retrieved chunks and their
provenance, so you (or a reviewer) can sanity-check retrieval and then open the
cited PDF page to confirm the snippet is really there.

Usage (from backend/):
    python query.py "what is the gross NPA ratio?"
    python query.py "treasury segment revenue" --k 5
    python query.py "who audits the bank" --full      # show full chunk text
"""
from __future__ import annotations

import argparse
import sys

from rag.store import search

# Windows consoles default to cp1252, which can't print some OCR'd glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="The question to retrieve for")
    ap.add_argument("--k", type=int, default=5, help="How many chunks to show")
    ap.add_argument("--full", action="store_true", help="Show full chunk text")
    ap.add_argument(
        "--no-hybrid", action="store_true",
        help="Force pure vector search (skip BM25 fusion) — for before/after comparison",
    )
    rr = ap.add_mutually_exclusive_group()
    rr.add_argument(
        "--rerank", action="store_true",
        help="Force the cross-encoder re-ranking stage ON (default: follow USE_RERANKER)",
    )
    rr.add_argument(
        "--no-rerank", action="store_true",
        help="Force the cross-encoder re-ranking stage OFF (default: follow USE_RERANKER)",
    )
    return ap


def search_options(args: argparse.Namespace) -> dict:
    """Map CLI flags to search() kwargs. With no flag, `rerank=None` so search()
    follows CONFIG.use_reranker — the same behaviour as the API."""
    rerank = True if args.rerank else (False if args.no_rerank else None)
    return {"top_k": args.k, "hybrid": False if args.no_hybrid else None, "rerank": rerank}


def main() -> None:
    args = build_parser().parse_args()

    results = search(args.question, **search_options(args))
    if not results:
        print("No results (is the index built? run `python ingest.py`).")
        return

    print(f'\nQuery: "{args.question}"\n' + "=" * 70)
    for i, r in enumerate(results, 1):
        print(f"\n#{i}  similarity={r.similarity:.3f}  ->  {r.doc}  page {r.page}  ({r.source})")
        text = r.text if args.full else " ".join(r.text.split())[:280] + " ..."
        print(text)


if __name__ == "__main__":
    main()
