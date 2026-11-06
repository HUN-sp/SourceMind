"""Vector store: exact cosine search over a NumPy matrix (local, no native deps),
optionally fused with a BM25 keyword ranking (hybrid retrieval).

Why not Chroma / FAISS / an HNSW index?
  Our corpus is tens of pages -> a few hundred chunks. Approximate-nearest-
  neighbour indexes (HNSW, IVF) exist to make search sublinear at the scale of
  millions of vectors. At a few hundred vectors, a single matrix multiply is
  already sub-millisecond AND exact (no recall loss from approximation). It also
  has ZERO native dependencies, so it builds on any reviewer's machine without a
  C++ toolchain (Chroma's hnswlib needs one on Windows). If the corpus grew to
  100k+ chunks I'd switch to FAISS; at this scale, simpler is correct.

Why ALSO keyword search (hybrid)?
  Pure vector search ranks by *meaning*, which fails on a specific class of our
  questions: a terse numeric row like "Capital Adequacy Ratio 19.88% ..." embeds
  poorly against a natural-language question, while chatty boilerplate that
  happens to echo the question's date/company words ("...HDFC BANK ... YEAR ENDED
  MARCH 31, 2026") scores high. Measured: the real answer row fell to rank #118.
  BM25 is the opposite — it rewards the literal rare phrase ("adequacy") that
  only the answer row contains. We fuse the two rankings with Reciprocal Rank
  Fusion (RRF) so a chunk that's strong in EITHER signal surfaces. BM25 is built
  in-memory from chunks.json on first search (pure-Python rank_bm25, no native
  deps, sub-ms at our scale), so the persisted index format is unchanged.

Persistence is just two files in INDEX_DIR:
  embeddings.npy  -> float32 matrix (N x d), L2-normalized
  chunks.json     -> the N chunk records (text + provenance), aligned by row
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from .chunk import Chunk, embedding_text
from .config import CONFIG
from .embed import embed_documents, embed_query

class IndexNotBuiltError(FileNotFoundError):
    """The search index doesn't exist yet (ingest hasn't been run)."""


class IndexCorruptError(IndexNotBuiltError):
    """The index files exist but don't agree with each other (e.g. an interrupted
    ingest). Treated like 'not built': the fix is to re-run ingest."""


_EMB_FILE = "embeddings.npy"
_META_FILE = "chunks.json"


@dataclass
class Retrieved:
    doc: str
    page: int
    text: str
    source: str
    similarity: float  # 0..1, higher = more relevant
    # metadata for citations + period/statement grounding
    doc_label: str = ""
    statement_type: str = ""
    fiscal_period: str = ""
    chunk_type: str = ""
    chunk_index: int = -1


def index_chunks(chunks: list[Chunk], batch_size: int = 128) -> int:
    """(Re)build the index from scratch and persist it to INDEX_DIR."""
    if not chunks:
        raise ValueError("no chunks to index (did extraction find any text?)")
    CONFIG.index_dir.mkdir(parents=True, exist_ok=True)

    vectors: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors.extend(embed_documents([embedding_text(c) for c in batch]))

    emb = np.asarray(vectors, dtype=np.float32)
    # Write to temp files, then rename into place, so a crash mid-write can never
    # leave a half-written file where the live index was. (The two renames are not
    # jointly atomic; _load() additionally verifies the pair agrees.)
    emb_tmp = CONFIG.index_dir / (_EMB_FILE + ".tmp")
    meta_tmp = CONFIG.index_dir / (_META_FILE + ".tmp")
    try:
        with open(emb_tmp, "wb") as f:
            np.save(f, emb)
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, ensure_ascii=False)
        os.replace(meta_tmp, CONFIG.index_dir / _META_FILE)
        os.replace(emb_tmp, CONFIG.index_dir / _EMB_FILE)
    finally:
        for tmp in (emb_tmp, meta_tmp):
            tmp.unlink(missing_ok=True)
    return len(chunks)


_cache: tuple[np.ndarray, list[dict]] | None = None
_bm25_cache = None  # lazily-built BM25 index, aligned row-for-row with _cache's meta


def _load() -> tuple[np.ndarray, list[dict]]:
    global _cache
    if _cache is None:
        emb_path = CONFIG.index_dir / _EMB_FILE
        meta_path = CONFIG.index_dir / _META_FILE
        if not emb_path.exists():
            raise IndexNotBuiltError(
                f"No index at {CONFIG.index_dir}. Run `python ingest.py` first."
            )
        try:
            emb = np.load(emb_path)
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:  # missing chunks.json, truncated/invalid files
            raise IndexCorruptError(
                f"Index at {CONFIG.index_dir} is unreadable ({e}). Re-run `python ingest.py`."
            ) from e
        if emb.ndim != 2 or emb.shape[0] != len(meta):
            raise IndexCorruptError(
                f"Index at {CONFIG.index_dir} is inconsistent ({emb.shape[0] if emb.ndim else 0} "
                f"embeddings vs {len(meta)} chunks). Re-run `python ingest.py`."
            )
        _cache = (emb, meta)
    return _cache


_generation = 0  # bumped on every reload so answer caches can tell the index changed


def generation() -> int:
    return _generation


def reload_index() -> None:
    """Drop the in-memory caches (used after re-ingesting)."""
    global _cache, _bm25_cache, _generation
    _cache = None
    _bm25_cache = None
    _generation += 1


# --- BM25 keyword index (hybrid retrieval) ---------------------------------
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. Numbers like '19.88%' split to '19','88' —
    fine, because the discriminating signal is the rare label words ('adequacy')."""
    return _WORD_RE.findall(text.lower())


# Common words + bare numbers carry no keyword signal and actively hurt BM25: a
# question like "...as of March 31, 2026" otherwise matches every page header
# (which all say "...ENDED MARCH 31, 2026"), burying the real answer row. We strip
# them from the QUERY only (chunks are still indexed on all their tokens).
_STOPWORDS = frozenset(
    "a an the of for to in on at and or as is are was were be been being it its this that these those "
    "what which who whom whose how when where why do does did have has had will would can could should "
    "by with from your you we our us their his her not no into about over per than then there here".split()
)


def _query_tokens(question: str) -> list[str]:
    content = [t for t in _tokenize(question) if t not in _STOPWORDS and not t.isdigit()]
    return content or _tokenize(question)  # never return empty


# --- Recency-aware tiebreak ------------------------------------------------
_MONTHS = (
    "january february march april may june july august "
    "september october november december"
).split()
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS)}
_PERIOD_RE = re.compile(r"(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),?\s*(\d{4})", re.I)


def _period_key(period: str) -> int:
    """Turn a fiscal_period like 'March 31, 2026' into a sortable int (20260331).
    Returns 0 when the period is unknown/unparseable."""
    m = _PERIOD_RE.search(period or "")
    if not m:
        return 0
    return int(m.group(3)) * 10000 + _MONTH_NUM[m.group(1).lower()] * 100 + int(m.group(2))


# --- Explicit periods / statement type named in the question -------------------
# "...as of March 31, 2026", "FY2026", "Q2 FY2026 (July-September 2025)". We resolve
# these to the quarter-end dates filings are labelled with, and (a) give chunks of a
# matching filing a small score bonus, (b) strip the expression from the text used for
# semantic/keyword matching — there it only matches the doc-label prefix that every
# chunk of a filing shares, drowning out the metric the user actually asked about.
_MONTH_ALIASES = {
    **{m: i + 1 for i, m in enumerate(_MONTHS)},
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MON = r"(" + "|".join(sorted(_MONTH_ALIASES, key=len, reverse=True)) + r")\.?"
_YEAR = r"((?:19|20)\d{2})"
_FYNUM = r"'?((?:19|20)\d{2}|\d{2})"
_QUARTER_END = {1: (6, 30, 0), 2: (9, 30, 0), 3: (12, 31, 0), 4: (3, 31, 1)}  # Indian FY: Apr-Mar


def _fy_year(g: str) -> int:
    y = int(g)
    return y if y >= 100 else 2000 + y


def _month_end(year: int, month: int) -> int:
    import calendar

    return year * 10000 + month * 100 + calendar.monthrange(year, month)[1]


# (regex, resolver(match) -> list[int]) — applied in order; matched text is removed.
_PERIOD_PATTERNS = [
    (re.compile(r"\b((?:q[1-4](?:\s*(?:,|&|/|\band\b|\bor\b))*\s*)+)(?:of\s*)?(?:fy|f\.y\.?)\s*" + _FYNUM + r"\b", re.I),  # Q2, Q3 and Q4 of FY2026
     lambda m: [
         (_fy_year(m.group(2)) - 1 + _QUARTER_END[int(q)][2]) * 10000 + _QUARTER_END[int(q)][0] * 100 + _QUARTER_END[int(q)][1]
         for q in re.findall(r"q([1-4])", m.group(1), re.I)
     ]),
    (re.compile(r"\b(?:fy|f\.y\.?|fiscal\s+year|financial\s+year)\s*" + _FYNUM + r"\b", re.I),
     lambda m: [_fy_year(m.group(1)) * 10000 + 331]),
    (re.compile(r"\b" + _MON + r"\s*(?:-|–|to)\s*" + _MON + r",?\s*" + _YEAR + r"\b", re.I),   # July-September 2025
     lambda m: [_month_end(int(m.group(3)), _MONTH_ALIASES[m.group(2).lower()])]),
    (re.compile(r"\b" + _MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+" + _YEAR + r"\b", re.I),   # March 31, 2026
     lambda m: [int(m.group(3)) * 10000 + _MONTH_ALIASES[m.group(1).lower()] * 100 + int(m.group(2))]),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MON + r",?\s+" + _YEAR + r"\b", re.I),   # 31 March 2026
     lambda m: [int(m.group(3)) * 10000 + _MONTH_ALIASES[m.group(2).lower()] * 100 + int(m.group(1))]),
    (re.compile(r"\b" + _MON + r",?\s+" + _YEAR + r"\b", re.I),                               # March 2026
     lambda m: [_month_end(int(m.group(2)), _MONTH_ALIASES[m.group(1).lower()])]),
    (re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-]" + _YEAR + r"\b"),                           # 31.03.2026
     lambda m: [int(m.group(3)) * 10000 + int(m.group(2)) * 100 + int(m.group(1))]),
]


def parse_query_periods(question: str) -> tuple[set[int], str]:
    """Resolve periods named in the question to sortable dates (YYYYMMDD, matching
    `_period_key`) and return (dates, question-with-those-expressions-removed)."""
    keys: set[int] = set()
    text = question
    for rx, resolve in _PERIOD_PATTERNS:
        def _sub(m, resolve=resolve):
            keys.update(resolve(m))
            return " "
        text = rx.sub(_sub, text)
    text = re.sub(r"\(\s*\)", " ", text)           # parentheses emptied by the removal
    text = " ".join(text.split())
    return keys, (text if keys else question)


def query_statement(question: str) -> str | None:
    """'consolidated' / 'standalone' if the question names exactly one, else None."""
    q = question.lower()
    cons = bool(re.search(r"\b(consolidated|group)\b", q))
    stand = bool(re.search(r"\bstandalone\b", q))
    if cons == stand:
        return None
    return "consolidated" if cons else "standalone"


def _query_names_period(question: str) -> bool:
    """True if the question pins a specific period (a year, or a month name). When
    it does, we must NOT bias toward recency — the explicit match should decide."""
    q = question.lower()
    if re.search(r"\b(19|20)\d{2}\b", q):  # a 4-digit year
        return True
    return any(mo in q for mo in _MONTHS)


def _recency_bonus(candidate_idx: np.ndarray, meta: list[dict], weight: float) -> dict[int, float]:
    """Small additive bonus per chunk, scaled by how recent its filing is among the
    candidates: the most recent period gets +weight, the oldest +0."""
    keys = {int(i): _period_key(meta[int(i)].get("fiscal_period", "")) for i in candidate_idx}
    distinct = sorted({k for k in keys.values() if k})
    if len(distinct) < 2:
        return {}
    rank = {k: r for r, k in enumerate(distinct)}  # oldest=0 ... newest=len-1
    span = len(distinct) - 1
    return {i: weight * (rank[k] / span) for i, k in keys.items() if k}


def _metadata_bonus(candidate_idx, meta, periods, statement, weight, base=None):
    """+weight for a chunk whose filing period is one the question named, and +weight
    for one whose statement type is the one it named. Added onto `base` bonuses."""
    out = dict(base or {})
    for i in candidate_idx:
        m = meta[int(i)]
        b = 0.0
        if periods and _period_key(m.get("fiscal_period", "")) in periods:
            b += weight
        if statement and m.get("statement_type") == statement:
            b += weight
        if b:
            out[int(i)] = out.get(int(i), 0.0) + b
    return out


# --- Cross-encoder re-ranking ----------------------------------------------
@lru_cache(maxsize=1)
def _reranker():
    """Load the cross-encoder once. Downloaded on first use (~80MB)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CONFIG.reranker_model)


_NUMERIC_Q = (
    "how much", "how many", "number of", "net profit", "profit", "income", "revenue",
    "ratio", "npa", "capital adequacy", "adequacy", "assets", "provision", "eps",
    "earnings per share", "dividend", "net worth", "deposits", "advances", "interest",
    "expense", "amount", "percentage", "crore", "what was the", "figure", "%",
)


def _wants_number(question: str) -> bool:
    """Heuristic: is the question asking for a financial figure? If so, we add a
    numeric-density vote so dense value rows beat fluent prose that merely mentions
    the metric. Prose questions (e.g. 'did the Dubai branch face action?') skip it."""
    q = question.lower()
    return any(kw in q for kw in _NUMERIC_Q)


def _numeric_density(text: str) -> float:
    """Fraction of tokens that contain a digit — high for table value rows, low for
    prose. Used to favour the row that actually carries the number."""
    toks = _tokenize(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if any(c.isdigit() for c in t)) / len(toks)


def _rerank(question: str, pool_idxs: list[int], meta: list[dict], top_k: int) -> list[int]:
    """Numeric-aware re-ranking by RANK FUSION (not pure cross-encoder override).

    `pool_idxs` arrives in FIRST-STAGE order (best first). We build up to three
    rankings and fuse them with RRF, then keep top_k:
      1. first-stage order (hybrid: vector + BM25) — already good for value rows,
      2. cross-encoder order — great at judging real relevance, BUT prone to
         preferring fluent prose over terse numeric rows,
      3. numeric-density order — ONLY for figure questions, so the row that holds
         the number can't be buried by a prose decoy.
    Fusing keeps the reranker's upside (it can still rescue a chunk first-stage
    missed) while stopping it from single-handedly burying the answer row."""
    scores = np.asarray(_reranker().predict([[question, meta[int(i)]["text"]] for i in pool_idxs]), dtype=float)
    rerank_order = [pool_idxs[int(j)] for j in np.argsort(-scores)]

    rankings = [list(pool_idxs), rerank_order]
    if _wants_number(question):
        dens = np.array([_numeric_density(meta[int(i)]["text"]) for i in pool_idxs])
        rankings.append([pool_idxs[int(j)] for j in np.argsort(-dens)])

    k = CONFIG.rerank_rrf_k
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]


_MIN_CHUNKS_FOR_FOCUS = 50


@lru_cache(maxsize=8)
def _common_terms(generation_: int, ratio: float) -> frozenset[str]:
    """Terms present in more than `ratio` of the PAGE-LEVEL chunks (e.g. 'bank', 'year',
    'ended'). Corpus-derived, so it adapts to whatever documents are indexed — no
    hand-written list. Row chunks are excluded: they're short and numerous, so counting
    them would dilute the ratio of terms that only appear in page text. Keyed on the
    index generation."""
    _, meta = _load()
    pages = [m for m in meta if m.get("chunk_type") != "row"]
    n = len(pages)
    if n < _MIN_CHUNKS_FOR_FOCUS:  # document frequency is meaningless on a tiny corpus
        return frozenset()
    df: dict[str, int] = {}
    for m in pages:
        for t in set(_tokenize(m["text"])):
            df[t] = df.get(t, 0) + 1
    return frozenset(t for t, c in df.items() if c / n > ratio)


def focus_query(question: str) -> str:
    """The question reduced to terms that discriminate between chunks: no stopwords,
    bare numbers, single characters or corpus-ubiquitous terms. Used for BOTH the
    semantic and keyword rankings — words every chunk shares ("HDFC Bank", "year
    ended") otherwise outweigh the metric ("net profit") and pull in letterhead and
    boilerplate chunks. Falls back to the original text if nothing would be left."""
    if not CONFIG.focus_query:
        return question
    common = _common_terms(_generation, CONFIG.common_term_ratio)
    kept = [
        t for t in _tokenize(question)
        if t not in _STOPWORDS and not t.isdigit() and len(t) > 1 and t not in common
    ]
    return " ".join(kept) if kept else question


def _get_bm25():
    """Build (once) a BM25 index over the chunk texts, aligned with meta order."""
    global _bm25_cache
    if _bm25_cache is None:
        from rank_bm25 import BM25Okapi

        _, meta = _load()
        _bm25_cache = BM25Okapi([_tokenize(m["text"]) for m in meta])
    return _bm25_cache


def _rrf_fuse(
    ranked_lists: list[np.ndarray],
    candidate_idx: np.ndarray,
    top_k: int,
    bonus: dict[int, float] | None = None,
    weights: list[float] | None = None,
) -> list[int]:
    """Reciprocal Rank Fusion: score each candidate by sum(1/(rrf_k + rank)) over
    every ranking it appears in, then take the best top_k. A chunk that ranks
    well in EITHER the vector or the keyword list rises to the top. `bonus` adds a
    small per-chunk score (e.g. a recency tiebreak)."""
    k = CONFIG.rrf_k
    scores: dict[int, float] = {int(i): 0.0 for i in candidate_idx}
    for n, ranking in enumerate(ranked_lists):
        w = weights[n] if weights else 1.0
        for rank, idx in enumerate(ranking):
            scores[int(idx)] += w / (k + rank)
    if bonus:
        for idx, b in bonus.items():
            if idx in scores:
                scores[idx] += b
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return ordered[:top_k]


def search(
    question: str,
    top_k: int | None = None,
    doc: str | None = None,
    hybrid: bool | None = None,
    rerank: bool | None = None,
) -> list[Retrieved]:
    """Retrieve the most relevant chunks, most-relevant first.

    Pipeline: (1) first-stage ranking — vector search, optionally fused with a
    BM25 keyword ranking via RRF (`hybrid`, the default) so literal label matches
    aren't buried; (2) optional cross-encoder re-ranking (`rerank`, the default)
    of a candidate pool down to top_k. `doc` optionally restricts to one document
    (the Phase 5 'filters' stretch). `hybrid=False` / `rerank=False` disable those
    stages — used by tests and for before/after comparison.

    Note: the `similarity` field on each result is ALWAYS the true cosine score
    (so the UI's "% match" stays honest), regardless of the fused/reranked order.
    """
    top_k = top_k or CONFIG.top_k
    hybrid = CONFIG.use_hybrid if hybrid is None else hybrid
    rerank = CONFIG.use_reranker if rerank is None else rerank
    # When re-ranking, first-stage gathers a larger candidate pool to choose from.
    pool_k = max(CONFIG.rerank_candidates, top_k) if rerank else top_k
    emb, meta = _load()

    # Separate WHAT was asked from WHEN: search on the metric words only, and use the
    # named period / statement type as a soft bonus on chunk metadata (see above).
    periods, search_text = parse_query_periods(question)
    statement = query_statement(question)
    search_text = focus_query(search_text)
    q = np.asarray(embed_query(search_text), dtype=np.float32)
    sims = emb @ q  # both sides are L2-normalized -> dot product == cosine

    # Optional document filter -> the candidate set we rank within.
    if doc is not None:
        candidate_idx = np.where(np.array([m["doc"] == doc for m in meta]))[0]
    else:
        candidate_idx = np.arange(len(meta))

    if candidate_idx.size == 0:
        return []

    # Vector ranking over the candidates (best first).
    vec_order = candidate_idx[np.argsort(-sims[candidate_idx])]

    if hybrid:
        # Keyword ranking over the same candidates, then fuse the two by RRF.
        bm25_scores = _get_bm25().get_scores(_query_tokens(search_text))
        bm25_order = candidate_idx[np.argsort(-bm25_scores[candidate_idx])]
        # Recency tiebreak: only when the question names no specific period.
        bonus = None
        if CONFIG.recency_boost > 0 and not _query_names_period(question):
            bonus = _recency_bonus(candidate_idx, meta, CONFIG.recency_boost)
        if CONFIG.period_boost > 0 and (periods or statement):
            bonus = _metadata_bonus(candidate_idx, meta, periods, statement, CONFIG.period_boost, bonus)
        # Rank row chunks and page-level chunks in SEPARATE lists. Their cosines live on
        # different scales (a short label-only row vs a long page chunk), so in one mixed
        # list the page chunks crowd the rows out regardless of relevance.
        is_row = np.array([m.get("chunk_type") == "row" for m in meta])
        # Row lists carry CONFIG.row_weight (default 1.0 = equal footing with page lists).
        lists, weights = [], []
        for ranked in (vec_order, bm25_order):
            for want_rows in (True, False):
                part = ranked[is_row[ranked] == want_rows]
                if part.size:
                    lists.append(part)
                    weights.append(CONFIG.row_weight if want_rows else 1.0)
        chosen = _rrf_fuse(lists, candidate_idx, pool_k, bonus, weights)
    else:
        chosen = vec_order[:pool_k].tolist()

    # Second stage: cross-encoder re-ranks the candidate pool down to top_k.
    if rerank and len(chosen) > 1:
        chosen = _rerank(question, chosen, meta, top_k)
    else:
        chosen = chosen[:top_k]

    out: list[Retrieved] = []
    for idx in chosen:
        m = meta[int(idx)]
        out.append(
            Retrieved(
                doc=m["doc"],
                page=int(m["page"]),
                text=m["text"],
                source=m.get("source", ""),
                similarity=float(sims[int(idx)]),
                doc_label=m.get("doc_label", ""),
                statement_type=m.get("statement_type", ""),
                fiscal_period=m.get("fiscal_period", ""),
                chunk_type=m.get("chunk_type", ""),
                chunk_index=int(m.get("chunk_index", -1)),
            )
        )
    return out


def list_documents() -> list[dict]:
    """One summary row per indexed document (sorted by name): label, period, size.
    Empty list if the index isn't built."""
    try:
        _, meta = _load()
    except IndexNotBuiltError:
        return []
    docs: dict[str, dict] = {}
    pages: dict[str, set[int]] = {}
    for m in meta:
        d = docs.setdefault(m["doc"], {
            "doc": m["doc"], "doc_label": m.get("doc_label", ""),
            "fiscal_period": m.get("fiscal_period", ""), "chunks": 0,
        })
        d["chunks"] += 1
        pages.setdefault(m["doc"], set()).add(int(m["page"]))
    for name, d in docs.items():
        d["pages"] = len(pages[name])
    return [docs[k] for k in sorted(docs)]


def collection_size() -> int:
    try:
        return len(_load()[1])
    except IndexNotBuiltError:  # not built, or corrupt
        return 0
