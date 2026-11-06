import { useEffect, useRef, useState } from "react";
import { askQuestion, getHealth, listDocuments } from "./api.js";
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
  const [documents, setDocuments] = useState([]);
  const [doc, setDoc] = useState(""); // "" = search all documents
  const [setupProblems, setSetupProblems] = useState([]);
  const abortRef = useRef(null); // AbortController of the in-flight request

  // Don't leave a request running if the component goes away.
  useEffect(() => () => abortRef.current?.abort(), []);

  // On load, ask the backend whether it is actually ready to answer, so setup
  // problems are explained up front instead of after the first failed question.
  useEffect(() => {
    let alive = true;
    getHealth().then((h) => {
      if (!alive) return;
      if (h === null) {
        setSetupProblems([
          "Can't reach the SourceMind backend. Start it with `uvicorn app:app --port 8000` from the backend/ folder.",
        ]);
      } else if (!h.ready) {
        setSetupProblems(h.problems ?? []);
      }
    });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let alive = true;
    listDocuments().then((d) => alive && setDocuments(d));
    return () => {
      alive = false;
    };
  }, []);

  async function submit(q) {
    const text = (q ?? question).trim();
    if (!text || loading) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await askQuestion(text, { doc: doc || undefined, signal: ctrl.signal });
      setResult(data);
    } catch (e) {
      if (e?.name === "AbortError") return; // user pressed Cancel: not an error
      setError(e.message || "Something went wrong.");
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      setLoading(false);
    }
  }

  function cancel() {
    abortRef.current?.abort();
  }

  function onSubmit(e) {
    e.preventDefault();
    submit();
  }

  function runExample(ex) {
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

      {setupProblems.length > 0 && (
        <div className="setup-banner" role="status" data-testid="setup-banner">
          <strong>Setup needed</strong>
          <ul>
            {setupProblems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      <form className="ask-form" onSubmit={onSubmit}>
        <textarea
          className="question-input"
          placeholder="Ask a question about the documents…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            // isComposing: Enter that confirms an IME candidate must not submit.
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent?.isComposing && !e.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          rows={3}
        />
        <button className="ask-button" type="submit" disabled={loading || !question.trim()}>
          {loading ? "Searching…" : "Ask"}
        </button>
        {loading && (
          <button className="cancel-button" type="button" onClick={cancel}>
            Cancel
          </button>
        )}
      </form>

      {documents.length > 0 && (
        <label className="doc-picker">
          <span>Search in</span>
          <select value={doc} onChange={(e) => setDoc(e.target.value)} disabled={loading}>
            <option value="">All documents</option>
            {documents.map((d) => (
              <option key={d.doc} value={d.doc}>
                {d.doc_label ? `${d.doc_label} (${d.doc})` : d.doc}
              </option>
            ))}
          </select>
        </label>
      )}

      <div className="examples">
        <span className="examples-label">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="example-chip"
            onClick={() => runExample(ex)}
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
