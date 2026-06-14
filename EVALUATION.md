# EVALUATION.md

Ten questions we put to our own system, with the **answer it produced**, the
**sources it cited**, and our **honest verdict** — including the ones it gets
wrong. Results are from live end-to-end runs against the backend (Groq
`llama-3.3-70b-versatile`). Required mix: includes unanswerable questions (the
system should decline) and paraphrased questions (worded differently from the
documents).

> Where retrieval is fragile, results can vary slightly between runs (LLM
> non-determinism + borderline ranking). We note those cases honestly rather than
> cherry-picking a good run.

---

### Q1 — Direct lookup ✅
**Q:** What is the gross NPA ratio as a percentage of gross advances?
**Answer:** "The gross NPA ratio as a percentage of gross advances is **1.15%** as
of March 31, 2026."
**Sources:** `data_4.pdf` p1 — Standalone Results — March 31, 2026 (sim 0.69)
**Verdict:** Correct. Right document (latest filing), right page, right figure and
period. Well grounded.

### Q2 — Paraphrase ✅
**Q:** How much bad debt does the bank have relative to its total lending?
**Answer:** "The bank has **1.15%** of its gross advances as gross NPAs and
**0.38%** of its net advances as net NPAs, as of March 31, 2026."
**Sources:** `data_4.pdf` p1 — Standalone Results — March 31, 2026 (sim 0.62)
**Verdict:** Correct, and notable: "bad debt relative to lending" shares **no
keywords** with "Non-Performing Assets / gross advances." Pure semantic match —
the embedding model did its job. Lower similarity (0.62 vs 0.69) but still retrieved.

### Q3 — Multi-period comparison ✅
**Q:** What were the bank's gross NPAs as of September 30, 2025, and how did they
compare to June 30, 2025?
**Answer:** "Gross NPAs as of September 30, 2025 were **34289.48**, a decrease from
**37040.80** as of June 30, 2025."
**Sources:** `data_2.pdf` p1 (Sep 2025) + `data_1.pdf` p1 (Jun 2025)
**Verdict:** Correct. Pulled the right figure from two different filings and
compared them. Good cross-document retrieval and column reading.

### Q4 — Paraphrase / multi-hop ✅
**Q:** Did the Dubai branch face any regulatory action?
**Answer:** "Yes, the Dubai branch (DIFC) received a decision notice from the Dubai
Financial Services Authority prohibiting it from taking on new customers; existing
customers are unaffected."
**Sources:** `data_4.pdf` p15 — Consolidated Results — March 31, 2026 (sim 0.69)
**Verdict:** Correct. "Dubai branch" → "DIFC Branch", "regulatory action" →
"decision notice / prohibition." Multi-hop paraphrase handled well.

### Q5 — Disclosure lookup ✅
**Q:** What did the bank set aside to cover potential future loan losses?
**Answer:** "A **floating provision of ₹9,000.00 crore**, during the year ended
March 31, 2026 (also reported for the nine months ended December 31, 2025)."
**Sources:** `data_3.pdf` p4 + `data_4.pdf` p15
**Verdict:** Correct. "Set aside for future loan losses" → "floating provision."
Note kept intact through chunking.

### Q6 — Cross-quarter comparison ✅
**Q:** How did the Gross NPA ratio change from September 30, 2025 to March 31, 2026?
**Answer:** "Decreased from **1.24%** (Sep 30, 2025) to **1.15%** (Mar 31, 2026) —
a 0.09 percentage-point decrease."
**Sources:** `data_2.pdf` p1 + `data_4.pdf` p1
**Verdict:** Correct. Both figures retrieved from their respective filings.

### Q7 — Unanswerable (out-of-domain) ✅
**Q:** Who is the CEO of Microsoft?
**Answer:** "I couldn't find this in the provided documents, so I can't answer it."
**Sources:** none.
**Verdict:** Correct refusal. Out-of-domain; the system declined rather than
answering from training knowledge.

### Q8 — Unanswerable (training-data trap) ✅
**Q:** What is the total number of HDFC Bank branches across India as of March 2026?
**Answer:** "I couldn't find this in the provided documents, so I can't answer it."
**Sources:** none.
**Verdict:** Correct refusal — the important one. Branch count (~9,000+) is widely
known from training data and tempting to hallucinate, but it isn't in the filings,
so the system declined. Grounding held. (Stock price and "top 10 borrowers"
likewise correctly declined.)

### Q9 — Direct value-row lookup ❌ (known weakness)
**Q:** What was the net profit of HDFC Bank for the full year ended March 31, 2026?
**Answer (wrong):** It reported the **OFS net gain on sale of shares (₹9,179.40
crore)** instead of the net profit (**₹74,671.29 crore**, standalone).
**Sources:** `data_4.pdf` p6 (standalone) + p15 — *not* the P&L row on p1.
**Verdict:** **Wrong.** This is our documented dense-value-row weakness
([DECISIONS.md §3](DECISIONS.md)): the terse "Net Profit for the period" row ranks
below fluent prose that semantically resembles "net profit" (a "net gain" note).
The correct value is in the index (`data_4` p1) — retrieval just doesn't surface
it for this phrasing. This is the single most important thing left to improve;
the fix is finer chunking of the P&L/ratios block.

### Q10 — Direct lookup, false refusal ❌
**Q:** What final dividend per share did HDFC Bank's Board propose for FY2026?
**Answer:** "I couldn't find this in the provided documents, so I can't answer it."
**Sources:** none.
**Verdict:** **False refusal.** The proposed final dividend is disclosed in a note
of the FY2026 filing, but the note either didn't extract cleanly from the scan or
its chunk didn't rank into the context, so the LLM saw nothing to answer from and
declined. Honest behaviour given what it retrieved, but the wrong outcome — same
root cause as Q9 (retrieval of a specific dense/note value), not a grounding flaw.

---

## Summary

| Behaviour | Count |
|---|---|
| Correct answer (grounded + cited) | 6 |
| Correct refusal (genuinely not in corpus) | 2 |
| **Wrong answer** (value row mis-retrieved) | 1 |
| **False refusal** (answer present but not retrieved) | 1 |

**Where it's strong:** paraphrase/semantic retrieval, multi-period and
cross-document comparisons, and honest refusal on out-of-domain questions
(including the training-data traps). When the right chunk is retrieved, grounding
and citation are reliable.

**Where it fails, and why:** both failures are the **same retrieval weakness** —
terse numeric value rows (net profit, a dividend note) ranking below prose, so the
LLM never sees the answer chunk. It is *not* a grounding failure (the system never
fabricates; it either cites a wrong-but-real chunk or honestly declines) and *not*
specific to the OCR'd document. The fix is finer-grained chunking of dense
financial tables — documented in DECISIONS.md as the top remaining item.
