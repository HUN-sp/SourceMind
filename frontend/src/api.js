// The single place the frontend talks to the backend.
// Calls go to "/api/ask", which Vite proxies to the FastAPI server (see vite.config.js).

export async function askQuestion(question) {
  let res;
  try {
    res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch (e) {
    // Network-level failure (backend not running, connection refused, etc.)
    throw new Error(
      "Could not reach the server. Is the backend running on port 8000?"
    );
  }

  if (!res.ok) {
    // The backend returns a helpful message in `detail` (e.g. missing API key).
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }

  return res.json(); // { answer, sources: [...], refused }
}
