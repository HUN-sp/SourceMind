"""Build the searchable index from the PDF corpus.

Usage (from the backend/ directory):
    python ingest.py                 # ingest everything in ./data
    python ingest.py --data-dir path # ingest a different folder
    python ingest.py --stats         # just print chunk stats, don't re-index

This is intentionally a one-shot offline step: parse -> chunk -> embed -> store.
The API never ingests on the fly; it only reads the index this script builds.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from rag.chunk import chunk_pages, count_tokens
from rag.config import CONFIG
from rag.extract import extract_corpus
from rag.store import index_chunks


def run(data_dir: Path, *, stats_only: bool) -> int:
    """Run the ingest pipeline. Returns a process exit code (0 ok, 1 error)."""
    t0 = time.time()
    print(f"[1/3] Extracting text from PDFs in {data_dir} ...")
    try:
        pages = extract_corpus(data_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    by_doc = Counter(p.doc for p in pages)
    by_source = Counter(p.source for p in pages)
    print(f"      {len(pages)} pages across {len(by_doc)} documents")
    for doc, n in sorted(by_doc.items()):
        print(f"        - {doc}: {n} pages")
    print(f"      extraction source: {dict(by_source)}")

    print("[2/3] Chunking ...")
    chunks = chunk_pages(pages)
    if not chunks:
        print("ERROR: extraction produced pages but no chunks.", file=sys.stderr)
        return 1
    tok_counts = [count_tokens(c.text) for c in chunks]
    print(
        f"      {len(chunks)} chunks | tokens/chunk "
        f"min={min(tok_counts)} avg={sum(tok_counts)//len(tok_counts)} max={max(tok_counts)}"
    )

    if stats_only:
        print("[stats] --stats given; skipping indexing.")
        return 0

    print("[3/3] Embedding + indexing ...")
    n = index_chunks(chunks)
    print(f"      indexed {n} chunks at {CONFIG.index_dir}")
    print(f"Done in {time.time() - t0:.1f}s.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest PDFs into the vector index.")
    ap.add_argument("--data-dir", default=None, help="Folder of PDFs (default: ./data)")
    ap.add_argument("--stats", action="store_true", help="Print stats only; no indexing")
    args = ap.parse_args()

    # Show "Skipping <file>" warnings from the extractor.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_dir = CONFIG.data_dir if args.data_dir is None else Path(args.data_dir)
    sys.exit(run(data_dir, stats_only=args.stats))


if __name__ == "__main__":
    main()
