Q 8  PASS   found @#1
Q 9  PASS   found @#1
Q10  PASS   found @#1
Q11  PASS   found @#5
Q12  PASS   1.24@#1, 1.15@#1
Q13  PASS   found @#1
Q14  FAIL   18641.28=MISS, 19221.05=MISS  (needle approx.)
Q15  FAIL   0 FY2026-notes chunks in top-8
Q16  PASS   found @#1
Q17  FAIL   MISS
Q18  DECLINE-expected   top-sim=0.75   (LLM-gated, not a retrieval test)
Q19  DECLINE-expected   top-sim=0.66   (LLM-gated, not a retrieval test)
Q20  DECLINE-expected   top-sim=0.78   (LLM-gated, not a retrieval test)

========================================================
RETRIEVAL: 10/17 answerable questions have the answer chunk in top-8
(declines are LLM-gated and reported above, not counted here)
PS C:\Users\HP\OneDrive\Desktop\thriftStone\backend> python - <<'PY'
>> import re
>> import numpy as np
>> from rag.store import _load
>> from rag.embed import embed_query
>>
>> # Load embeddings and metadata
>> emb, meta = _load()
>>
>> # Normalize text: lowercase + remove non-alphanumeric chars
>> sq = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
>>
>> def rank_each(query, marker, label):
>>     sims = emb @ np.asarray(embed_query(query), dtype=np.float32)
>>     order = list(np.argsort(-sims))
>>
>>     print("\n" + "=" * 80)
>>     print(f"{label}: {query!r}")
>>     print("=" * 80)
>>
>>     for doc in ["data_1.pdf", "data_2.pdf", "data_3.pdf", "data_4.pdf"]:
>>         hit = next(
>>             (
>>                 (
>>                     rank,
>>                     float(sims[i]),
>>                     meta[i]["page"],
>>                     meta[i].get("source")
>>                 )
>>                 for rank, i in enumerate(order, start=1)
>>                 if meta[i]["doc"] == doc and marker(meta[i]["text"])
>>             ),
>>             None
>>         )
>>
>>         if hit:
>>             rank, sim, page, source = hit
>>             print(
>>                 f"{doc:<12} "
>>                 f"rank #{rank:>3}   "
>>                 f"sim={sim:.3f}   "
>>                 f"page={page}   "
>>                 f"source={source}"
>>             )
>>         else:
>>             print(f"{doc:<12} none")
>>
>> # Test 1: Net Profit
>> rank_each(
>>     "What was the net profit for the period?",
>>     lambda t: "netprofitfortheperiod" in sq(t),
>>     "NET PROFIT"
>> )
>>
>> # Test 2: Capital Adequacy Ratio
>> rank_each(
>>     "What is the Capital Adequacy Ratio?",
>>     lambda t: (
>>         "capitaladequacyratio" in sq(t)
>>         and re.search(r"1[0-9]\.[0-9]", t)
>>     ),
>>     "CAPITAL ADEQUACY RATIO"
>> )
>>
>> PY
At line:1 char:11
+ python - <<'PY'
+           ~
Missing file specification after redirection operator.
At line:1 char:10
+ python - <<'PY'
+          ~
The '<' operator is reserved for future use.
At line:1 char:11
+ python - <<'PY'
+           ~
The '<' operator is reserved for future use.
At line:4 char:1
+ from rag.store import _load
+ ~~~~
The 'from' keyword is not supported in this version of the language.
At line:5 char:1
At line:8 char:19
+ emb, meta = _load()
+                   ~
An expression was expected after '('.
At line:11 char:35
+ sq = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
At line:8 char:19
At line:8 char:19
At line:8 char:19
+ emb, meta = _load()
+                   ~
An expression was expected after '('.
At line:8 char:19
+ emb, meta = _load()
+                   ~
An expression was expected after '('.
At line:8 char:19
+ emb, meta = _load()
+                   ~
An expression was expected after '('.
At line:8 char:19
+ emb, meta = _load()
+                   ~
An expression was expected after '('.
At line:8 char:19
+ emb, meta = _load()
+                   ~
An expression was expected after '('.
At line:8 char:19
+ emb, meta = _load()
+                   ~
At line:8 char:19
+ emb, meta = _load()
+                   ~
An expression was expected after '('.
At line:11 char:35
Not all parse errors were reported.  Correct the reported errors and try again.
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : MissingFileSpecification

PS C:\Users\HP\OneDrive\Desktop\thriftStone\backend> python .\test_rank.py   

================================================================================
NET PROFIT: 'What was the net profit for the period?'
================================================================================
data_1.pdf   rank # 10 sim=0.607 page=1 source=text-layer
data_2.pdf   rank #  3 sim=0.622 page=8 source=text-layer
data_3.pdf   rank # 19 sim=0.593 page=7 source=text-layer
data_4.pdf   rank #  1 sim=0.625 page=11 source=ocr

================================================================================
CAPITAL ADEQUACY RATIO: 'What is the Capital Adequacy Ratio?'
================================================================================
data_1.pdf   rank # 51 sim=0.549 page=1 source=text-layer
data_2.pdf   rank # 92 sim=0.538 page=1 source=text-layer
data_3.pdf   rank # 73 sim=0.542 page=1 source=text-layer
data_4.pdf   rank # 36 sim=0.557 page=1 source=ocr
PS C:\Users\HP\OneDrive\Desktop\thriftStone\backend> python query.py "What is the capital adequacy ratio as of March 31, 2026?" --k 8

Query: "What is the capital adequacy ratio as of March 31, 2026?"
======================================================================

#1  similarity=0.701  ->  data_4.pdf  page 1  (ocr)
Standalone Results — March 31, 2026 Quarterended Yearended 31.03.2026 31.12.2026 31.03.2025 31.03.2026 31.03.2026 NPA Ratios: (a) Gross NPAs 34061.19 35178.98 35222.64 34061.19 35222.64 （b)Net NPAs 11169.54 11981.75 11320.43 11169.54 11320.43 (c)% of Gross NPAs to Gross Advances  ...

#2  similarity=0.659  ->  data_4.pdf  page 1  (ocr)
Standalone Results — March 31, 2026 Quarterended Yearended 31.03.2026 31.12.2026 31.03.2025 31.03.2026 31.03.2026 11 Tax Expense(Refernote 18) 5972.30 5606.19 5727.51 20497.40 21130.70 12 Net Profit from ordlnary actlvities after tax{10)-(11) 19221.05 18663.75 17616.14 74671.29 6 ...

#3  similarity=0.697  ->  data_4.pdf  page 16  (ocr)
Consolidated Results — March 31, 2026 together referred to as “"the Group") for the year ended 31 March 2026, attached herewith, being results: include the annual financial results of the entities mentioned in Annexure I; b are presented in accordance with the requirements of Reg ...

#4  similarity=0.688  ->  data_1.pdf  page 3  (text-layer)
Standalone Results — June 30, 2025 Particulars 30.06.2025 30.06.2024 31.03.2025 2024. 6 The Board of Directors at its meeting held on July 19, 2025, declared a special interim dividend of? 5.00 per equity share pre-bonus issuance. Effect of the dividend has been reckoned in deter ...

#5  similarity=0.658  ->  data_4.pdf  page 2  (ocr)
Standalone Results — March 31, 2026 Quarterended Yearended 31.03.2026 31.12.2025 31.03.2025 31.03.2026 1247937.97 1499816.13 1247937.97 Other Banking Operations 112867.13 111587.23 112358.81 112867.13 112358.81 Unallocated 21386.51 22751.33 24137.77 21386.51 24137.77 Tota! 436488 ...

#6  similarity=0.716  ->  data_4.pdf  page 12  (ocr)
Consolidated Results — March 31, 2026 Quarter ended Year ended 31.03.2026 31.12.2025 31.03.2026 31.03.2025 2528642.55 2416040.26 2312429.69 2528642.55 2312429.69 Wholesale Banking 1130276.17 988303.80 956136.34 1130276.17 956136.34 Other Banking Operations 3997.87 3956.36 8513.18 ...

#7  similarity=0.694  ->  data_4.pdf  page 11  (ocr)
Consolidated Results — March 31, 2026 CONSOLIDATEDFINANCIALRESULTSFORTHEQUARTERANDYEARENDEDMARCH31,2026 Operating profit before provisions and contingencies (3)-(6) 31111.68 30581.81 29378.75 128797.67 110416.66 8 Provisions(other thantax)andcontingencies(Refernote9) 3440.05 3620 ...

#8  similarity=0.684  ->  data_4.pdf  page 4  (ocr)
Standalone Results — March 31, 2026 theyearended March31,2026havebeen subjected toan audit by the joint statutory auditors of the Bankviz.Batlboi&Purohit,Chartered Accountants and S R& Co.LLP,Charlered Accountants.The financial resuils for year ended March 31,2025 were audited by ...
PS C:\Users\HP\OneDrive\Desktop\thriftStone\backend> """Token-free retrieval harness for the 20-question bank.

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
