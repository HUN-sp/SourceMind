import SourceCard from "./SourceCard.jsx";

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
      </section>
    );
  }

  return (
    <section className="answer-panel">
      <h2 className="answer-label">Answer</h2>
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
    </section>
  );
}
