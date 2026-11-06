// The single place the frontend talks to the backend.
// Calls go to "/api/ask", which Vite proxies to the FastAPI server (see vite.config.js).

const DEFAULT_TIMEOUT_MS = 60_000;

// FastAPI returns `detail` as a string for our own errors but as a LIST of
// validation objects for 422s. Turn either into one readable sentence.
function describeDetail(detail, status) {
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    const msg = detail.map((d) => d?.msg).filter(Boolean)[0];
    if (msg) return `Please check your question: ${msg}.`;
  }
  if (status === 429) return "Too many requests. Please wait a moment and try again.";
  if (status >= 500) {
    // No JSON detail: usually the dev proxy answering for a backend that is down.
    return `The server isn't responding properly (HTTP ${status}). Is the backend running on port 8000?`;
  }
  return `Request failed (${status}).`;
}

/**
 * Ask a question.
 * @param {string} question
 * @param {{doc?: string, timeoutMs?: number, signal?: AbortSignal}} [opts]
 *   doc       restrict the search to one document (file name)
 *   timeoutMs give up after this long instead of spinning forever
 *   signal    lets the caller cancel; cancellation rejects with an AbortError
 */
export async function askQuestion(question, { doc, timeoutMs = DEFAULT_TIMEOUT_MS, signal } = {}) {
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, timeoutMs);
  const relayAbort = () => ctrl.abort();
  if (signal) {
    if (signal.aborted) ctrl.abort();
    else signal.addEventListener("abort", relayAbort, { once: true });
  }

  try {
    let res;
    try {
      res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(doc ? { question, doc } : { question }),
        signal: ctrl.signal,
      });
    } catch (e) {
      if (timedOut) {
        throw new Error(
          `The request took too long (over ${Math.round(timeoutMs / 1000)}s). The server may be busy — please try again.`
        );
      }
      if (e?.name === "AbortError") throw e; // cancelled by the caller
      // Network-level failure (backend not running, connection refused, etc.)
      throw new Error("Could not reach the server. Is the backend running on port 8000?");
    }

    if (!res.ok) {
      let body = null;
      try {
        body = await res.json();
      } catch {
        /* response had no JSON body */
      }
      throw new Error(describeDetail(body?.detail, res.status));
    }

    return await res.json(); // { answer, sources: [...], refused }
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", relayAbort);
  }
}

/**
 * The indexed documents, for the "Search in" picker. Never throws: the picker is a
 * convenience, so on any failure we return [] and the UI simply hides it.
 */
export async function listDocuments() {
  try {
    const res = await fetch("/api/documents");
    if (!res.ok) return [];
    const body = await res.json();
    return Array.isArray(body?.documents) ? body.documents : [];
  } catch {
    return [];
  }
}

/**
 * Backend readiness from /health: `{ready, problems[], ...}`, or `null` if the
 * backend can't be reached (network error, or the dev proxy's bare 5xx).
 */
export async function getHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}
