"""Token-free retrieval harness for the 20-question bank.

Runs each question through store.search() ONLY (embeddings + BM25 + optional
reranker — all local, ZERO Groq tokens) and checks whether the chunk holding the
known answer lands in the top-k. This is our fast feedback loop for chunk-selection
without spending the LLM quota.

Pass = the answer-bearing chunk is in the top-k.
For multi-source questions, every required figure must appear (in any top-k chunk).
For must-decline questions, we just report the top similarity (LLM-gated, not a
retrieval pass/fail).

Usage:
    python retrieval_harness.py            # uses CONFIG (reranker off by default)
    python retrieval_harness.py --rerank   # force reranker on, to A/B
"""
from __future__ import annotations

import sys

from rag.store import reload_index, search

TOP_K = 8

# any  -> pass if ANY needle appears in some top-k chunk
# all  -> pass only if EVERY needle appears (across top-k) — multi-source questions
# notes-> pass if >= n chunks come from data_4 pages 4-6 (the FY2026 notes)
# decline -> informational only (report top similarity)
QUESTIONS = [
    {"n": 1,  "q": "What was the net profit of HDFC Bank for the full year ended March 31, 2026?", "any": ["74671.29"]},
    {"n": 2,  "q": "What is HDFC Bank's Capital Adequacy Ratio as of March 31, 2026?", "any": ["19.71"]},
    {"n": 3,  "q": "When did HDFC Bank issue bonus shares and in what proportion?", "any": ["bonus equity share", "1:1", "1: 1", "proportion of 1"]},
    {"n": 4,  "q": "What final dividend per share did HDFC Bank's Board propose for FY2026?", "any": ["dividend"], "weak": True},
    {"n": 5,  "q": "How much money did the bank make after paying taxes in the most recent financial year?", "any": ["74671.29"]},
    {"n": 6,  "q": "What fraction of the bank's loans have gone bad?", "any": ["1.15%", "Gross NPAs to Gross Advances"]},
    {"n": 7,  "q": "Did the Dubai branch face any regulatory action?", "any": ["DIFC", "DFSA"]},
    {"n": 8,  "q": "What did the bank set aside to cover potential future loan losses?", "any": ["floating provision"]},
    {"n": 9,  "q": "What was HDFC Bank's Gross NPA amount as of March 31, 2026, and how does it compare to March 31, 2025?", "any": ["34061.19"]},
    {"n": 10, "q": "What were HDFC Bank's total segment assets in the Retail Banking segment as of March 31, 2026?", "any": ["Retail Banking"]},
    {"n": 11, "q": "In the COVID-19 resolution framework disclosure, how many personal loan accounts were classified as Standard consequent to restructuring as of March 31, 2026?", "any": ["resolution framework"]},
    {"n": 12, "q": "How did HDFC Bank's Gross NPA ratio change from September 30, 2025 to March 31, 2026?", "all": ["1.24", "1.15"]},
    {"n": 13, "q": "What was HDFC Bank's interest income for Q2 FY2026 (July-September 2025)?", "any": ["76690.70"]},
    {"n": 14, "q": "How did HDFC Bank's standalone net profit trend across Q2, Q3, and Q4 of FY2026?", "all": ["18641.28", "19221.05"], "weak": True},
    {"n": 15, "q": "Summarise all the key events or corporate actions that HDFC Bank disclosed in its FY2026 annual results notes.", "notes": 2},
    {"n": 16, "q": "What did HDFC Bank say about the impact of the new Labour Codes on its financials?", "any": ["Labour Codes"]},
    {"n": 17, "q": "What was HDFC Bank Group's (consolidated) net profit for FY2026?", "any": ["79219.46"]},
    {"n": 18, "q": "What is HDFC Bank's current stock price?", "decline": True},
    {"n": 19, "q": "Who are HDFC Bank's top 10 corporate borrowers?", "decline": True},
    {"n": 20, "q": "What is the total number of HDFC Bank branches across India as of March 2026?", "decline": True},
]


def _norm(s: str) -> str:
    return " ".join(s.split())


def main(rerank: bool | None) -> None:
    reload_index()
    passed = total = 0
    for item in QUESTIONS:
        res = search(item["q"], top_k=TOP_K, rerank=rerank)
        low = [_norm(r.text).lower() for r in res]

        if item.get("decline"):
            mx = max((r.similarity for r in res), default=0.0)
            print(f"Q{item['n']:>2}  DECLINE-expected   top-sim={mx:.2f}   (LLM-gated, not a retrieval test)")
            continue

        total += 1
        if "notes" in item:
            cnt = sum(1 for r in res if r.doc == "data_4.pdf" and r.page in (4, 5, 6))
            ok = cnt >= item["notes"]
            detail = f"{cnt} FY2026-notes chunks in top-{TOP_K}"
        elif "all" in item:
            found = {nd: next((j + 1 for j, t in enumerate(low) if nd.lower() in t), None) for nd in item["all"]}
            ok = all(v for v in found.values())
            detail = ", ".join(f"{nd}@#{r}" if r else f"{nd}=MISS" for nd, r in found.items())
        else:
            rank = next((j + 1 for j, t in enumerate(low) for nd in item["any"] if nd.lower() in t), None)
            ok = rank is not None
            detail = f"found @#{rank}" if rank else "MISS"

        passed += ok
        flag = "PASS" if ok else "FAIL"
        weak = "  (needle approx.)" if item.get("weak") else ""
        print(f"Q{item['n']:>2}  {flag}   {detail}{weak}")

    print(f"\n{'='*56}\nRETRIEVAL: {passed}/{total} answerable questions have the answer chunk in top-{TOP_K}")
    print("(declines are LLM-gated and reported above, not counted here)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rr = True if "--rerank" in sys.argv else (False if "--no-rerank" in sys.argv else None)
    main(rr)
