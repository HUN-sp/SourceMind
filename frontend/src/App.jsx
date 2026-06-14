import { useState } from "react";
import { askQuestion } from "./api.js";
import AnswerPanel from "./components/AnswerPanel.jsx";

const EXAMPLES = [
  "What is the gross NPA ratio?",
  "Who are the statutory auditors of the bank?",
  "What is the capital adequacy ratio?",
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function submit(q) {
    const text = (q ?? question).trim();
    if (!text || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await askQuestion(text);
      setResult(data);
    } catch (e) {
      setError(e.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e) {
    e.preventDefault();
    submit();
  }

  function useExample(ex) {
    setQuestion(ex);
    submit(ex);
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>SourceMind</h1>
        <p className="tagline">
          Answers grounded in the provided documents — with the source for every
          answer. If it isn&apos;t in the documents, the system says so.
        </p>
      </header>

      <form className="ask-form" onSubmit={onSubmit}>
        <textarea
          className="question-input"
          placeholder="Ask a question about the documents…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={3}
        />
        <button className="ask-button" type="submit" disabled={loading || !question.trim()}>
          {loading ? "Searching…" : "Ask"}
        </button>
      </form>

      <div className="examples">
        <span className="examples-label">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="example-chip"
            onClick={() => useExample(ex)}
            disabled={loading}
          >
            {ex}
          </button>
        ))}
      </div>

      <main className="results">
        {loading && (
          <div className="status loading">
            <span className="spinner" aria-hidden="true" />
            Retrieving passages and composing a grounded answer…
          </div>
        )}
        {error && !loading && (
          <div className="status error" role="alert">
            <strong>Error:</strong> {error}
          </div>
        )}
        {result && !loading && <AnswerPanel result={result} />}
      </main>
    </div>
  );
}
