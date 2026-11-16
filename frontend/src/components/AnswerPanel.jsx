import { useState } from "react";
import SourceCard from "./SourceCard.jsx";

// Plain-text version of an answer + its citations, for the clipboard.
export function formatForCopy(result) {
  const lines = result.sources.map((s) => {
    const name = s.doc_label ? `${s.doc_label} (${s.doc})` : s.doc;
    return `- ${name}, page ${s.page}`;
  });
  return lines.length
    ? `${result.answer}\n\nSources:\n${lines.join("\n")}`
    : result.answer;
}

function CopyButton({ result }) {
  const [state, setState] = useState("idle"); // idle | copied | failed

  async function copy() {
    try {
      await navigator.clipboard.writeText(formatForCopy(result));
      setState("copied");
    } catch {
      setState("failed");
    }
    setTimeout(() => setState("idle"), 2000);
  }

  const label = { idle: "Copy answer", copied: "Copied ✓", failed: "Copy failed" }[state];
  return (
    <button type="button" className="copy-button" onClick={copy}>
      {label}
    </button>
  );
}

// Where the response came from and what it cost — part of making the system legible.
export function ResponseMeta({ result }) {
  if (result.latency_ms === undefined || result.latency_ms === null) return null;
  const gate1 = result.refused && (result.top_similarity ?? 1) < (result.min_similarity ?? 0);
  let how;
  if (result.cached) how = result.preset ? "pre-computed demo response · 0 LLM tokens" : "answer cache · 0 LLM tokens";
  else if (gate1) how = "refused at the similarity gate · 0 LLM tokens";
  else how = "live retrieval + LLM";
  return (
    <p className="response-meta" data-testid="response-meta">
      {result.latency_ms} ms · {how}
    </p>
  );
}

// Renders the result of an /ask call. Two visual modes:
//   - refused: the system honestly didn't find the answer in the documents.
//   - answered: show the grounded answer + the sources that support it.
export default function AnswerPanel({ result }) {
  if (result.refused) {
    return (
      <section className="answer-panel refused">
        <h2 className="answer-label">Not found in the documents</h2>
        <p className="answer-text">{result.answer}</p>
        <p className="refused-note">
          The system only answers from the provided documents. It declined rather
          than guess.
        </p>
        <ResponseMeta result={result} />
      </section>
    );
  }

  return (
    <section className="answer-panel">
      <div className="answer-head">
        <h2 className="answer-label">Answer</h2>
        {result.cached && (
          <span className="cached-badge" title="Served from memory — no model call was made">
            Instant answer
          </span>
        )}
        <CopyButton result={result} />
      </div>
      {/* Plain-text render — React escapes it, so no HTML injection is possible. */}
      <p className="answer-text">{result.answer}</p>

      <h3 className="sources-label">
        Sources <span className="sources-count">({result.sources.length})</span>
      </h3>
      <ul className="source-list">
        {result.sources.map((s, i) => (
          <SourceCard key={`${s.doc}-${s.page}-${i}`} source={s} index={i} />
        ))}
      </ul>
      <ResponseMeta result={result} />
    </section>
  );
}
