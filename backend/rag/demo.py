"""Public-demo support: preset questions, their pre-computed responses, and the
evaluation summary shown in the UI's "Retrieval & Evals" inspector.

Why pre-computed responses: the three one-click demo chips are what nearly every
visitor clicks first. Serving them from a JSON file means they load instantly and
spend ZERO LLM tokens, so the Groq free-tier quota is reserved for visitors' own
questions. The file is produced by running the REAL pipeline once
(`python build_demo_cache.py`), never hand-written, and it records a fingerprint
of the index it was built against: if the server's index differs, the cache is
ignored and the presets are answered live instead (never stale citations).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .config import CONFIG

log = logging.getLogger(__name__)

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
CACHE_FILE = DEMO_DIR / "demo_cache.json"
EVALS_FILE = DEMO_DIR / "evals.json"

# The three behaviours a reviewer should see in under a minute.
PRESETS: list[dict] = [
    {
        "id": "numeric",
        "kind": "Answerable · exact figure",
        "label": "Gross NPA ratio, Mar 2026",
        "question": "What was HDFC Bank's Gross NPA ratio (gross NPAs as a percentage of "
                    "gross advances) as of March 31, 2026?",
        "shows": "An exact figure with the document and page it came from.",
    },
    {
        "id": "multihop",
        "kind": "Multi-hop · cross-filing",
        "label": "NPA trend Sep 2025 → Mar 2026",
        "question": "How did HDFC Bank's Gross NPA ratio change from September 30, 2025 "
                    "to March 31, 2026?",
        "shows": "Two figures from two different quarterly filings, combined into one grounded answer.",
    },
    {
        "id": "trap",
        "kind": "Unanswerable · trap",
        "label": "Crypto holdings in 2026?",
        "question": "How much cryptocurrency did HDFC Bank hold on its balance sheet as of "
                    "March 31, 2026?",
        "shows": "Calibrated abstention: the filings never mention crypto, so it declines instead of guessing.",
    },
]


def normalize(question: str) -> str:
    return " ".join(question.split()).casefold()


_BY_QUESTION = {normalize(p["question"]): p for p in PRESETS}


def preset_for(question: str, doc: str | None) -> dict | None:
    """The preset this exact question corresponds to (only when searching all docs)."""
    if doc is not None:
        return None
    return _BY_QUESTION.get(normalize(question))


def index_fingerprint() -> str:
    """sha256 of the index's chunks.json — identifies the exact corpus a cache was built on."""
    path = CONFIG.index_dir / "chunks.json"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def load_cache() -> dict[str, dict]:
    """{preset id: response dict}, or {} when the file is missing or built on another index."""
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if data.get("index_sha256") != index_fingerprint():
        log.warning("demo_cache.json was built on a different index; presets will be answered live. "
                    "Re-run `python build_demo_cache.py`.")
        return {}
    return data.get("responses", {})


def load_evals() -> dict:
    try:
        return json.loads(EVALS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
