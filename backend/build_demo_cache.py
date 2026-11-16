"""Pre-compute the demo chips' responses by running the REAL pipeline once.

    python build_demo_cache.py            # needs the built index + GROQ_API_KEY

Writes demo/demo_cache.json: for each preset in rag/demo.py, the exact /ask
response (answer, citations, retrieved chunks with BM25 / BGE / RRF scores) plus
a fingerprint of the index it was built on. The server serves these in a few ms
with zero LLM tokens; if the index changes the file is ignored until re-built.
Costs at most one LLM call per preset (a gate-1 refusal costs none).
"""
from __future__ import annotations

import json
import sys
import time

from app import _to_response
from rag import demo
from rag.pipeline import ask
from rag.store import collection_size


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # e.g. Windows cp1252 consoles vs U+202F in answers
    if collection_size() == 0:
        print("No index. Run `python ingest.py` first.", file=sys.stderr)
        return 1
    responses = {}
    for p in demo.PRESETS:
        t0 = time.perf_counter()
        resp = _to_response(ask(p["question"]))
        ms = round((time.perf_counter() - t0) * 1000)
        responses[p["id"]] = resp.model_dump(exclude={"cached", "preset", "latency_ms"})
        verdict = "REFUSED" if resp.refused else "ANSWERED"
        cites = ", ".join(f"{s.doc} p{s.page}" for s in resp.sources) or "-"
        print(f"[{p['id']}] {verdict} in {ms} ms | {resp.answer}\n    cites: {cites}")
    demo.DEMO_DIR.mkdir(exist_ok=True)
    payload = {"index_sha256": demo.index_fingerprint(), "indexed_chunks": collection_size(),
               "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "responses": responses}
    demo.CACHE_FILE.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {demo.CACHE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
