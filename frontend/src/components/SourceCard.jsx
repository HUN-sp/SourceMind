// One cited source: document name, page, and the snippet text.
// The snippet is rendered as plain text (React escapes it) — never as raw HTML —
// which satisfies the brief's "render the model's text safely" requirement.

export default function SourceCard({ source, index }) {
  return (
    <li className="source-card">
      <div className="source-head">
        <span className="source-index">[{index + 1}]</span>
        <span className="source-doc" title={source.doc}>
          {source.doc_label || source.doc}
        </span>
        <span className="source-page">page {source.page}</span>
      </div>
      {/* show the raw filename too, when we also have a human label */}
      {source.doc_label && <div className="source-file">{source.doc}</div>}
      <p className="source-snippet">{source.snippet}</p>
    </li>
  );
}
