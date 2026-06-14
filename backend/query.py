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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="The question to retrieve for")
    ap.add_argument("--k", type=int, default=5, help="How many chunks to show")
    ap.add_argument("--full", action="store_true", help="Show full chunk text")
    ap.add_argument(
        "--no-hybrid", action="store_true",
        help="Force pure vector search (skip BM25 fusion) — for before/after comparison",
    )
    ap.add_argument(
        "--no-rerank", action="store_true",
        help="Skip the cross-encoder re-ranking stage — for before/after comparison",
    )
    args = ap.parse_args()

    results = search(
        args.question, top_k=args.k,
        hybrid=not args.no_hybrid, rerank=not args.no_rerank,
    )
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
