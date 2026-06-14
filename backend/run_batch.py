"""Run the NEW questions from the 20-question bank against the live backend.

Paced for the per-minute rate limit; retries once on a transient limit; stops
cleanly if the per-DAY token cap is hit. Skips any question that still fails.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

API = "http://localhost:8000/ask"
GAP = 8           # seconds between calls (per-minute rate-limit courtesy)
RETRY_WAIT = 20   # seconds to wait once on a transient (per-minute) limit

# (num, question, expected) — only the ones NOT already done.
QUESTIONS = [
    (2,  "What is HDFC Bank's Capital Adequacy Ratio as of March 31, 2026?", "answer"),
    (1,  "What was the net profit of HDFC Bank for the full year ended March 31, 2026?", "answer"),
    (17, "What was HDFC Bank Group's (consolidated) net profit for FY2026?", "answer"),
    (18, "What is HDFC Bank's current stock price?", "decline"),
    (19, "Who are HDFC Bank's top 10 corporate borrowers?", "decline"),
    (20, "What is the total number of HDFC Bank branches across India as of March 2026?", "decline"),
    (5,  "How much money did the bank make after paying taxes in the most recent financial year?", "answer"),
    (7,  "Did the Dubai branch face any regulatory action?", "answer"),
    (12, "How did HDFC Bank's Gross NPA ratio change from September 30, 2025 to March 31, 2026?", "answer"),
    (13, "What was HDFC Bank's interest income for Q2 FY2026 (July-September 2025)?", "answer"),
    (14, "How did HDFC Bank's standalone net profit trend across Q2, Q3, and Q4 of FY2026?", "answer"),
    (10, "What were HDFC Bank's total segment assets in the Retail Banking segment as of March 31, 2026?", "answer"),
    (11, "In the COVID-19 resolution framework disclosure, how many personal loan accounts were classified as Standard consequent to restructuring as of March 31, 2026?", "answer"),
    (15, "Summarise all the key events or corporate actions that HDFC Bank disclosed in its FY2026 annual results notes.", "answer"),
    (16, "What did HDFC Bank say about the impact of the new Labour Codes on its financials?", "answer"),
]


def ask(question: str) -> dict:
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main() -> None:
    out = []
    for num, q, expected in QUESTIONS:
        result = None
        for attempt in range(2):
            try:
                result = ask(q)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")
                if "per day" in detail.lower() or "tpd" in detail.lower():
                    print(f"\nQ{num}: DAILY TOKEN CAP HIT — stopping here.\n  {detail[:160]}")
                    _save(out)
                    return
                if attempt == 0:
                    print(f"Q{num}: transient limit ({e.code}); waiting {RETRY_WAIT}s and retrying once...")
                    time.sleep(RETRY_WAIT)
                    continue
                print(f"Q{num}: SKIPPED after retry ({e.code})")
            except Exception as e:  # noqa: BLE001
                print(f"Q{num}: SKIPPED (error: {e})")
                break

        if result is None:
            out.append({"q": num, "question": q, "skipped": True})
            time.sleep(GAP)
            continue

        refused = result.get("refused", False)
        got = "decline" if refused else "answer"
        ok = "PASS" if got == expected else "FAIL"
        srcs = "; ".join(
            f"{s.get('doc_label') or s['doc']} p{s['page']} ({s['similarity']})"
            for s in result.get("sources", [])
        )
        print(f"\nQ{num} [{ok}] (expected {expected}, got {got})")
        print(f"  Q: {q}")
        print(f"  A: {result.get('answer','')[:300]}")
        if srcs:
            print(f"  sources: {srcs}")
        out.append({"q": num, "question": q, "expected": expected, "got": got, **result})
        time.sleep(GAP)

    _save(out)


def _save(out: list) -> None:
    with open("batch_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    answered = sum(1 for r in out if not r.get("skipped") and "got" in r)
    print(f"\n{'='*60}\nRan {answered} question(s). Saved -> backend/batch_results.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
