"""The grounded-answer pipeline: question -> {answer, sources, refused}.

This is where the assignment's #1 criterion lives: grounding & honesty. The flow:

  1. Retrieve the most relevant chunks (vector search).
  2. GROUNDING GATE #1 (cheap, pre-LLM): if nothing is even remotely relevant
     (best similarity < floor), refuse without calling the LLM.
  3. Build a numbered context block from the chunks, each tagged with its
     (document, page) provenance.
  4. Ask the LLM to answer ONLY from that block, in strict JSON, and to report
     which passages it used — or to flag `not_covered`.
  5. GROUNDING GATE #2 (the LLM's own judgment): if it says not_covered, refuse.
  6. Map the passages the LLM used back to their provenance -> citations.

Two gates, because neither alone is enough: the score floor catches "nothing
relevant retrieved", the LLM flag catches "retrieved something, but it doesn't
actually answer the question".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .config import CONFIG
from .generate import generate as _default_generate
from .store import Retrieved, search

REFUSAL_MESSAGE = (
    "I couldn't find this in the provided documents, so I can't answer it."
)

SYSTEM_PROMPT = """You are a careful question-answering assistant for a set of documents.

Rules you must follow:
- Answer ONLY using the numbered passages provided. Do NOT use any outside or prior knowledge.
- Quote figures exactly as they appear. Cite the passages you used by their numbers.
- Write the answer as a short, direct sentence (or two). Do NOT dump a bare list
  of every number you see — give the figure a human would want.

WHEN TO ANSWER vs REFUSE (read this carefully — do NOT over-refuse):
- These passages are extracted from SCANNED financial tables, so the text is often
  noisy: words may be merged, letters mis-read, and columns slightly misaligned.
  This messiness is EXPECTED. Do NOT refuse because the text looks messy or because
  you are not 100% certain which column a number belongs to.
- If ANY passage contains a figure that matches the metric being asked about,
  you MUST answer with it. Pick the best-matching value, give it, and briefly note
  the period/statement it came from (you may flag any uncertainty in one short
  clause). Answering with a well-cited best estimate is REQUIRED — refusing is wrong.
- Set "not_covered": true ONLY when NO passage mentions the requested metric at
  all (e.g. an out-of-domain or off-topic question). Absence of the metric is the
  ONLY valid reason to refuse.

These are periodic financial statements, so the SAME figure is usually reported
across several period columns AND across several of the documents. Handle that
like an analyst, using the labels and headers we give you:
- Each passage is tagged with its document, page, statement type (standalone vs
  consolidated) and the period the filing covers, e.g.
  "[2] data_4.pdf, page 1 — Standalone Results — March 31, 2026 [table ...]".
- Table passages BEGIN with their column headers, e.g.
  "Quarter ended Year ended Particulars 31.03.2026 31.12.2025 31.03.2025 31.03.2026 31.03.2025".
  The numbers in a row line up with those columns in order. Use the headers to
  pick the value for the period the question asks about: for a FULL-YEAR question
  use the "Year ended" column whose date matches; for a quarter use the matching
  "Quarter ended" column.
- Prefer the passage whose statement type and period MATCH the question. If the
  question doesn't specify standalone vs consolidated, prefer standalone and say so.
- If the question does NOT specify a period, answer for the MOST RECENT period
  available — compare the period labels and dates and use the latest one.
- COMPARISON / CHANGE questions ("how did X change from A to B", "compare X across
  periods"): give the value for EACH period (with its date) and state the direction
  and size of the change. Do NOT refuse just because the two figures sit in
  different passages — that is exactly the kind of question you should piece
  together from multiple passages.
- For a single-value question, lead with the SINGLE most relevant value and state
  its document, statement type and period. You may note the figure is also reported
  for other periods, but do NOT list them all unless a comparison was asked.

Respond with a single JSON object and nothing else, in this exact shape:
{
  "not_covered": <true ONLY if NO passage mentions the requested metric, else false>,
  "answer": "<the concise grounded answer, or an empty string if not_covered>",
  "used_passages": [<the numbers of the passages you used, e.g. 1, 3>]
}"""


@dataclass
class Source:
    doc: str
    page: int
    snippet: str
    similarity: float
    doc_label: str = ""
    statement_type: str = ""
    fiscal_period: str = ""


@dataclass
class AskResult:
    answer: str
    sources: list[Source]
    refused: bool
    # transparency: what retrieval surfaced + the top score (handy for the UI/debug)
    retrieved: list[Retrieved]
    top_similarity: float


def build_context(chunks: list[Retrieved]) -> str:
    """Render retrieved chunks as a numbered, provenance- and metadata-tagged block."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        label = f" — {c.doc_label}" if c.doc_label else ""
        if c.chunk_type == "table":
            kind = " [table: each row's numbers line up with the period columns shown at the top]"
        else:
            kind = ""
        tag = f"[{i}] {c.doc}, page {c.page}{label}{kind}"
        blocks.append(f"{tag}\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[Retrieved]) -> str:
    return f"PASSAGES:\n{build_context(chunks)}\n\nQUESTION: {question}"


def parse_llm_json(raw: str) -> dict:
    """Parse the model's JSON. Tolerant of stray text around the object."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def ask(
    question: str,
    *,
    top_k: int | None = None,
    doc: str | None = None,
    generate: Callable[[str, str], str] = _default_generate,
) -> AskResult:
    """Answer a question, grounded in the corpus. `generate` is injectable for tests."""
    chunks = search(question, top_k=top_k, doc=doc)
    # Gate #1 asks "is anything relevant at all?", so it must look at the BEST
    # cosine in the set. Under hybrid (RRF) ordering chunks[0] is the fused top,
    # not necessarily the max-cosine chunk — so take the max explicitly.
    top_sim = max((c.similarity for c in chunks), default=0.0)

    # --- Gate #1: nothing remotely relevant -> refuse before spending an LLM call.
    if not chunks or top_sim < CONFIG.min_similarity:
        return AskResult(
            answer=REFUSAL_MESSAGE, sources=[], refused=True,
            retrieved=chunks, top_similarity=top_sim,
        )

    # --- Present passages in DOCUMENT order (doc, position-in-doc), not similarity
    # order. When an answer spans sequential chunks this keeps context coherent and
    # stops the model from confusing one period's column with another's.
    ordered = sorted(chunks, key=lambda c: (c.doc, c.chunk_index))

    # --- Ask the LLM, constrained to the retrieved passages.
    raw = generate(SYSTEM_PROMPT, build_user_prompt(question, ordered))
    try:
        data = parse_llm_json(raw)
    except (json.JSONDecodeError, ValueError):
        # Malformed output -> fail safe by refusing rather than guessing.
        return AskResult(
            answer=REFUSAL_MESSAGE, sources=[], refused=True,
            retrieved=chunks, top_similarity=top_sim,
        )

    # --- Gate #2: the LLM judged the passages insufficient.
    if data.get("not_covered") or not str(data.get("answer", "")).strip():
        return AskResult(
            answer=REFUSAL_MESSAGE, sources=[], refused=True,
            retrieved=chunks, top_similarity=top_sim,
        )

    # --- Map the passages the model actually used (numbered over `ordered`) back to
    # citations.
    used = data.get("used_passages") or []
    used_idx = {int(n) - 1 for n in used if str(n).strip().isdigit()}
    cited = [ordered[i] for i in sorted(used_idx) if 0 <= i < len(ordered)]
    if not cited:  # model answered but forgot to cite -> fall back to the best hit
        cited = [max(chunks, key=lambda c: c.similarity)]

    sources = [
        Source(
            doc=c.doc, page=c.page, snippet=c.text, similarity=c.similarity,
            doc_label=c.doc_label, statement_type=c.statement_type,
            fiscal_period=c.fiscal_period,
        )
        for c in cited
    ]
    return AskResult(
        answer=str(data["answer"]).strip(), sources=sources, refused=False,
        retrieved=chunks, top_similarity=top_sim,
    )
