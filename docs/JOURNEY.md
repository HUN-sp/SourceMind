# The Journey — what we hit, what we did, and why (plain English)

This file is written like I'm explaining it to a friend. No jargon. For each
thing: **what went wrong**, **how we fixed it**, and **the honest reason** we
made the choice we made. If someone asks "why did you do it this way?" — the
answer is here, in words you can actually say out loud.

First, the one-line goal so everything below makes sense:

> A user asks a question. We find the right bits of the PDFs, answer **only**
> from those bits, show *which page* the answer came from, and if the answer
> isn't in the documents, we **say so** instead of making something up.

---

## Part 1 — Getting the text out of the PDFs

### Challenge: the four PDFs are not the same kind of file
We opened the PDFs before writing code. They're all different:
- **data_2** — a normal digital PDF. The text is right there, easy to read.
- **data_1 and data_3** — these are *scans* (pictures of pages), but someone had
  already run text-recognition on them, so there's hidden text we can grab. Only
  the fancy logo bits come out as garbage.
- **data_4** — a pure scan. Just images. **Zero** real text inside. Like a photo
  of a page.

**How we approached it:** instead of writing special code for each file (the
brief bans that), we made one general rule:
> Try to read the text normally. If a page comes back almost empty, it must be a
> picture — so run image-to-text (OCR) on it instead.

That rule handled all four files, and it would handle a brand-new file the
reviewer swaps in too. We checked it works: 46 pages were read normally, 23 pages
(all of data_4) automatically went through OCR. That's our proof.

**Tradeoff — honest reason:** OCR isn't perfect. It sometimes glues words
together or misreads a letter (`Return on assets` → `Relurn on assels`). We chose
to live with that because **the numbers come through fine**, and trying to
perfectly clean it would be a lot of fragile code for little gain. We'd rather
show you the page so you can check it yourself.

### Challenge: garbled logo junk in the text
Scanned pages spit out nonsense lines from logos and headers (stuff like
`[' " l=1•1iil`).

**How we fixed it:** a simple rule — if a short line is mostly weird symbols
instead of normal letters, drop it. We do **not** hunt for specific words,
because that would be cheating (special-casing). It's a general "does this line
look like real text?" check.

---

## Part 2 — Cutting the text into pieces ("chunks")

### Why we cut it up at all
We can't search a whole 20-page document at once. We cut it into smaller pieces so
we can find the *exact* piece that answers a question.

### Challenge: a chunk must never mix two pages
If one chunk had text from page 4 **and** page 5, we couldn't honestly say "this
answer came from page 4." The whole point is showing the right page.

**How we fixed it:** a chunk always stays inside one page. One page becomes one or
more chunks — never the other way around.

### Challenge: pieces that are too big get silently cut off
The tool that turns text into searchable form can only handle so much at once. If
a chunk is too big, the end of it gets quietly thrown away — and that part becomes
impossible to find later. Scary, because you don't get an error.

**How we fixed it:** we measure each chunk in the *same units the tool uses* and
keep every chunk comfortably under the limit. We also let neighboring chunks
**overlap** a little, so if an important sentence sits right on the cut line, it
still shows up whole in at least one piece.

**Tradeoff — honest reason:** this overlap creates a couple of very tiny chunks.
We left them. Fixing it would mean extra merging code, and at our small size it
makes no real difference. Simpler is better here.

---

## Part 3 — Searching (finding the right pieces)

### How search works, in plain terms
We turn every chunk into a list of numbers that captures its *meaning*. We do the
same to the question. Then "most relevant" just means "whose numbers are closest
to the question's numbers." This is why you can ask in different words than the
document uses and still get a hit.

### Challenge: which search engine to use?
Most RAG tutorials reach for a fancy database (Chroma, FAISS). We tried Chroma
first — **and it wouldn't even install on Windows** because it needs a C++
compiler. A reviewer on Windows would hit the exact same wall.

**How we fixed it:** we do the math ourselves with a plain, dependency-free
library (NumPy). At our size (~224 pieces) this is *instant* and actually *more
accurate* (the fancy databases trade a little accuracy for speed at huge scale —
speed we don't need).

**Tradeoff — honest reason:** if we ever had millions of pieces, the simple
approach would get slow and we'd switch to FAISS. At a few hundred pieces, simple
is genuinely the *right* call, not a lazy one. That's the honest answer: "I used
the simplest thing that's correct at this scale, and I know exactly when I'd
change it."

---

## Part 4 — Answering honestly (the most important part)

The brief cares most about this: **don't make stuff up, and admit when you don't
know.** We use **two safety checks**, because one alone isn't enough.

### Safety check #1 — is anything even close?
Before we bother the AI, we look at the best match. If nothing is even remotely
related to the question, we refuse right away. Cheap and fast.

### Safety check #2 — the AI itself decides
We hand the AI *only* the pieces we found and tell it: "Answer using **only**
these. If they don't actually answer the question, say 'not covered.'" If it says
not covered, we refuse.

### Challenge: why do we need BOTH checks?
We tested it. Some nonsense questions *looked* related by the numbers (gibberish
scored higher than a real unanswerable question!). So the "is it close?" check
alone would let junk through. The AI reading the actual text is the smarter judge.
But the AI is slow and costs money, so the first cheap check saves us from even
calling it when nothing's there. **Two checks: one cheap and fast, one smart.**

### Real bugs we hit and fixed (good stories to tell)
- **Wrong period's number.** Asked for a ratio, it gave a real number — but from
  the wrong year's column. The tables have several years side by side and the date
  headers are scrambled. Fix: we told the AI to lead with the most recent value
  and name the document/page, instead of silently guessing a column.
- **A correct answer got refused.** The piece with the real answer was ranked
  #6, but we only looked at the top 5 — so we missed it and wrongly said "not
  found." Fix: we look at the top 8 now.
- **The sneaky one — old settings file.** We changed a setting in the code, but
  the app ignored it. Turns out a leftover settings file (`.env`) was overriding
  the code. Lesson: when behavior doesn't match the code, check the settings file
  first.
- **Number-soup answers.** It once dumped a list of 8 percentages from all the
  files — technically correct, totally useless. Fix: we told it to answer like a
  person would — one clear number, say where it's from, mention (not list) that
  older years exist.

### The honest weak spot we still have
If you ask about a **specific quarter/date**, we can sometimes pull the number
from the wrong filing. Why: the date headers are scrambled in the scans, and we
don't yet record each document's reporting date. The safety net is that we always
show *which* document we used, so you can catch it. The real fix is a stretch-goal
item (smarter search + document filters), which is exactly why we want Section 3
rock-solid first before going there.

---

## Part 5 — The website (frontend)

### What it does
A box to type a question, the answer shown clearly, the source pages shown right
underneath so you can verify, and proper "loading…" and "something broke"
messages.

### Challenge: browser security blocking the backend
Browsers block a page from talking to a server on a different port. Our page and
our backend are on different ports.

**How we fixed it:** the page just calls `/api/...` and a small proxy quietly
forwards it to the backend. To the browser it all looks like one place, so no
security complaints, and our code doesn't hardcode any web address.

### Safety: we never let document text run as code
We always show the AI's answer and the document snippets as **plain text**. Even
if a document contained sneaky HTML or script, it can't run. It's just shown as
words.

**Tradeoff — honest reason:** plain CSS, no fancy UI framework. The brief says it
*penalizes* over-engineered front-ends and rewards a clean, correct one. So we
kept it deliberately simple.

---

## Part 6 — Where we actually stand right now

**Working and checked:**
- Reads all four PDFs (including the all-images one) into searchable form.
- Finds the right pages and answers from them, with sources you can open.
- Honestly refuses when the answer isn't there.
- Clean handling when the key is missing, the network is down, or the AI returns junk.

**Still to do before we call "the core" finished (Section 3):**
- The **self-test write-up**: 8 questions we ask our own system (including one it
  *should* refuse and one worded differently than the docs), with our honest grade
  for each.
- The **paperwork the brief wants**: a README (how to run it), a DECISIONS file
  (the choices above, short), the self-test file, and a few small tests.

**The honest bottom line:** the actual app works end-to-end and we can defend
every choice in plain words. What's left before stretch goals is mostly *writing
down* what we did and adding a few tests — not building more features. Once that's
done, the natural next step is the one stretch item that fixes our real weak spot:
smarter search so specific-date questions hit the right filing.

---

## Part 7 — The search overhaul (fixing the wrong/missing answers)

We tested the system with real questions and found it was **refusing or answering
wrong** on things that were clearly in the documents (net profit, capital adequacy
ratio, total income). We dug in and found the root cause was **not the AI** — it
was how we were cutting up and finding the financial tables. Here's what we fixed,
in plain words.

### Problem: the table headers got separated from the numbers
A results table is big, so we cut it into pieces. The **column headers** ("Quarter
ended / Year ended" and the dates) ended up in one piece, and the **number rows**
("Net Profit ... 74671.29") in another. So when we found the number piece, it had
*no idea* which column was which year. **Fix:** we now detect the header row on a
page and **glue it back onto every piece** of that table. Now each number row
carries its own column labels.

### Problem: the scanned pages had glued-together words
The image-only document (data_4) came out of image-to-text with words fused, like
`AdequacyRatio` and `Quarterended`. Search couldn't match "adequacy" against
`adequacyratio`. **Fix:** a small cleanup step re-splits those fused words
(`AdequacyRatio` → `Adequacy Ratio`). General rule, no hardcoding.

### Problem: search ranked chatty text above the actual numbers
Plain meaning-based search liked wordy paragraphs that *mention* a topic over the
terse number rows that *answer* it. **Fix:** we added **two more search signals**:
- **Keyword search (BM25)** alongside meaning search, blended together — so the
  literal phrase "capital adequacy ratio" pulls its row up.
- **A re-ranker**: after we gather ~25 candidate pieces, a small model re-reads
  each one *together with the question* and re-orders them, keeping the best 8.
  This is much better at "does this piece actually answer the question."

### Problem: it answered for the wrong year when you didn't say a year
**Fix:** each piece now carries a **label** with the filing's statement type
(standalone vs consolidated) and its period ("March 31, 2026"). When a question
doesn't name a period, we gently **prefer the most recent filing**. When it does
name a period/month, we leave that off and let the exact match win.

### Problem: it refused too often on messy tables
The AI was treating "the text looks messy / I'm not 100% sure which column" as a
reason to refuse. **Fix:** we rewrote its instructions — messy scanned text is
*expected*; if any passage has a figure matching the metric, **answer it** (and
note the period). Only refuse when **no** passage mentions the metric at all. We
also told it how to handle **comparison** questions (give both periods + the change).

---

## Part 8 — We tested it (10 questions) and where it stands

We fired 10 varied questions at the running system. **8 out of 10 behaved as
expected** — a big jump from before. The wins: net profit, NPA, auditors,
comparisons, and paraphrased questions ("bad debt relative to lending") all answer
correctly now, and off-topic questions still get refused.

The **two honest limitations** that remain:

1. **The "(Basel III)" wording trips it up.** If you ask "capital adequacy ratio?"
   it answers fine — the real value rows come straight up. But the exact test
   wording *"capital adequacy ratio **(Basel III)**"* misfires: the words "Basel
   III" appear in the *disclosure notes*, not in the row with the actual number, so
   search gets pulled toward those notes. We chose **not** to special-case this —
   chasing one parenthetical would be overfitting, and the normal phrasing works.
   The capital adequacy ratio a bank reports *is* the Basel III ratio anyway.

2. **The free AI tier has a daily limit.** Groq's free plan allows ~100,000 tokens
   per day. Because each answer sends several big table chunks as context, a run of
   ~10 questions can exhaust the day's budget — after which requests come back as a
   clean "the model is unavailable" error (not a crash). For heavy testing you'd
   add a paid key or wait for the daily reset. This is an account limit, not a bug.
