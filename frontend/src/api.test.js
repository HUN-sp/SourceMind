import { afterEach, describe, expect, it, vi } from "vitest";
import { askQuestion } from "./api.js";

function jsonResponse(status, body, headers = {}) {
  return new Response(JSON.stringify(body), { status, headers });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("askQuestion", () => {
  it("posts the question (and optional doc) and returns the JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { answer: "x", sources: [], refused: false }));
    vi.stubGlobal("fetch", fetchMock);
    const out = await askQuestion("hi", { doc: "a.pdf" });
    expect(out.answer).toBe("x");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/ask");
    expect(JSON.parse(init.body)).toEqual({ question: "hi", doc: "a.pdf" });
  });

  it("omits doc when none is selected", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { answer: "x", sources: [], refused: false }));
    vi.stubGlobal("fetch", fetchMock);
    await askQuestion("hi");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ question: "hi" });
  });

  it("surfaces the backend's detail string", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(503, { detail: "Index missing" })));
    await expect(askQuestion("q")).rejects.toThrow("Index missing");
  });

  it("turns FastAPI's list-style 422 detail into a readable message (not [object Object])", async () => {
    const detail = [{ type: "string_too_long", loc: ["body", "question"], msg: "String should have at most 1000 characters" }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(422, { detail })));
    const err = await askQuestion("q").catch((e) => e);
    expect(err.message).not.toContain("[object Object]");
    expect(err.message).toMatch(/too long|at most 1000/i);
  });

  it("falls back to a generic message when the body isn't JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>bad gateway</html>", { status: 502 })));
    await expect(askQuestion("q")).rejects.toThrow(/502/);
  });

  it("reports an unreachable server clearly", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(askQuestion("q")).rejects.toThrow(/Could not reach the server/);
  });

  it("times out instead of spinning forever", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn((url, init) => new Promise((_, reject) => {
        init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }))
    );
    const p = askQuestion("q", { timeoutMs: 5000 }).catch((e) => e);
    await vi.advanceTimersByTimeAsync(5001);
    const err = await p;
    expect(err.message).toMatch(/took too long/i);
  });

  it("lets the caller cancel (AbortError is passed through, not shown as an error)", async () => {
    const ctrl = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn((url, init) => new Promise((_, reject) => {
        init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }))
    );
    const p = askQuestion("q", { signal: ctrl.signal }).catch((e) => e);
    ctrl.abort();
    const err = await p;
    expect(err.name).toBe("AbortError");
  });
});

import { listDocuments } from "./api.js";

describe("listDocuments", () => {
  it("returns the documents array", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, { documents: [{ doc: "a.pdf" }] })));
    expect(await listDocuments()).toEqual([{ doc: "a.pdf" }]);
  });

  it("returns [] on any failure (the picker is optional, never blocks asking)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("down")));
    expect(await listDocuments()).toEqual([]);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(500, {})));
    expect(await listDocuments()).toEqual([]);
  });
});

import { getHealth } from "./api.js";

describe("getHealth", () => {
  it("returns the health body", async () => {
    const body = { ready: true, problems: [] };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, body)));
    expect(await getHealth()).toEqual(body);
  });

  it("reports unreachable (null) on network failure or a proxy 5xx", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("down")));
    expect(await getHealth()).toBeNull();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("proxy error", { status: 500 })));
    expect(await getHealth()).toBeNull();
  });
});

describe("server errors without a detail", () => {
  it("name the likely cause for 5xx responses (Vite's proxy returns a bare 500 when the backend is down)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("", { status: 500 })));
    await expect(askQuestion("q")).rejects.toThrow(/backend running/i);
  });
});
