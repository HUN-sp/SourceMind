import { useState } from "react";

// "Retrieval & Evals" drawer: makes the system's evidence inspectable.
//   Retrieval tab — every chunk retrieval surfaced for the active question, with the
//                   score each first-stage signal gave it (BGE cosine, BM25) and the
//                   fused RRF score that decided the final order.
//   Evals tab     — the hand-graded evaluation and the reranker A/B ablation.

function fmt(n, digits) {
  return n === null || n === undefined ? "—" : Number(n).toFixed(digits);
}

function GateSummary({ result }) {
  const top = result.top_similarity ?? 0;
  const floor = result.min_similarity ?? 0.3;
  if (!result.retrieval?.length) return null;
  let verdict;
  if (top < floor) {
    verdict = "Gate 1 refused before any LLM call: nothing relevant enough was retrieved.";
  } else if (result.refused) {
    verdict =
      "Gate 1 passed (related passages exist), but at gate 2 the model judged that none of them " +
      "contains the answer, so it abstained instead of guessing.";
  } else {
    verdict = "Both gates passed; the answer cites the highlighted chunks below.";
  }
  return (
    <p className="gate-summary">
      Best BGE cosine <strong>{fmt(top, 3)}</strong> vs refusal floor <strong>{fmt(floor, 2)}</strong>.{" "}
      {verdict}
    </p>
  );
}

function RetrievalTab({ result }) {
  if (!result) {
    return <p className="inspector-empty">Ask a question (or click a sample above) to see what retrieval surfaced and why.</p>;
  }
  const rows = result.retrieval ?? [];
  if (!rows.length) return <p className="inspector-empty">No retrieval details for this response.</p>;
  const maxRrf = Math.max(...rows.map((r) => r.rrf ?? 0)) || 1;
  return (
    <>
      <GateSummary result={result} />
      <p className="inspector-legend">
        Ranked by <b>RRF</b> = Σ 1/(60 + rank) over the BM25 and BGE-small rankings (table rows and page
        chunks ranked separately), plus a small period/recency bonus. # = rank among all searched chunks.
      </p>
      <ol className="chunk-list" data-testid="retrieval-list">
        {rows.map((c, i) => (
          <li key={`${c.doc}-${c.page}-${i}`} className={`chunk-row${c.cited ? " cited" : ""}`}>
            <div className="chunk-head">
              <span className="chunk-rank">#{i + 1}</span>
              <span className="chunk-src" title={c.doc_label}>
                {c.doc} · p{c.page}
              </span>
              <span className="chunk-type">{c.chunk_type === "row" ? "table row" : "page"}</span>
              {c.cited && <span className="chunk-cited">cited</span>}
            </div>
            <div className="chunk-scores">
              <span className="score">
                <em>BGE cos</em> {fmt(c.cosine, 3)} <small>#{c.vec_rank ?? "—"}</small>
              </span>
              <span className="score">
                <em>BM25</em> {fmt(c.bm25, 2)} <small>#{c.bm25_rank ?? "—"}</small>
              </span>
              <span className="score rrf">
                <em>RRF</em> {fmt(c.rrf, 4)}
                <span className="rrf-bar" aria-hidden="true">
                  <span style={{ width: `${Math.max(4, ((c.rrf ?? 0) / maxRrf) * 100)}%` }} />
                </span>
              </span>
            </div>
            <details className="chunk-text">
              <summary>Chunk text</summary>
              <p>{c.snippet}</p>
            </details>
          </li>
        ))}
      </ol>
    </>
  );
}

function EvalsTab({ evals }) {
  if (!evals || !evals.eval10) return <p className="inspector-empty">Evaluation summary unavailable.</p>;
  const { eval10, ablation, pipeline } = evals;
  return (
    <div className="evals">
      {pipeline && (
        <section>
          <h4>Pipeline</h4>
          <ol className="pipeline-steps">
            {pipeline.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ol>
        </section>
      )}
      <section>
        <h4>{eval10.title}</h4>
        <p className="eval-note">{eval10.note}</p>
        <table className="eval-table">
          <tbody>
            {eval10.rows.map((r) => (
              <tr key={r.label}>
                <td>{r.label}</td>
                <td className="num">{r.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="eval-takeaway">{eval10.takeaway}</p>
      </section>
      {ablation && (
        <section>
          <h4>{ablation.title}</h4>
          <p className="eval-note">{ablation.note}</p>
          <table className="eval-table">
            <tbody>
              {ablation.rows.map((r) => (
                <tr key={r.label}>
                  <td>{r.label}</td>
                  <td className="num">{r.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {ablation.rank_shifts?.length > 0 && (
            <table className="eval-table shifts">
              <thead>
                <tr>
                  <th>Answer-chunk rank</th>
                  <th className="num">off</th>
                  <th className="num">on</th>
                </tr>
              </thead>
              <tbody>
                {ablation.rank_shifts.map((r) => (
                  <tr key={r.q}>
                    <td>{r.q}</td>
                    <td className="num">#{r.off}</td>
                    <td className={`num ${r.on > r.off ? "worse" : "better"}`}>#{r.on}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="eval-takeaway">{ablation.takeaway}</p>
        </section>
      )}
    </div>
  );
}

export default function Inspector({ result, evals }) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState("retrieval");
  return (
    <section className={`inspector${open ? " open" : ""}`}>
      <button
        type="button"
        className="inspector-toggle"
        aria-expanded={open}
        aria-controls="inspector-body"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="chev" aria-hidden="true">{open ? "▾" : "▸"}</span>
        Retrieval &amp; Evals
        <span className="inspector-hint">scores behind every answer</span>
      </button>
      {open && (
        <div id="inspector-body" className="inspector-body">
          <div className="tabs" role="tablist">
            {[
              ["retrieval", "Retrieval"],
              ["evals", "Evals & ablation"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={tab === id}
                className={`tab${tab === id ? " active" : ""}`}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </div>
          {tab === "retrieval" ? <RetrievalTab result={result} /> : <EvalsTab evals={evals} />}
        </div>
      )}
    </section>
  );
}
