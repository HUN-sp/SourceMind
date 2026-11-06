import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App.jsx";
import * as api from "./api.js";

beforeEach(() => {
  vi.spyOn(api, "listDocuments").mockResolvedValue([]);
  vi.spyOn(api, "getHealth").mockResolvedValue({ ready: true, problems: [] });
});
afterEach(() => vi.restoreAllMocks());

const DOCS = [
  { doc: "data_1.pdf", doc_label: "Standalone Results — June 30, 2025", fiscal_period: "June 30, 2025", pages: 3, chunks: 9 },
  { doc: "data_4.pdf", doc_label: "Standalone Results — March 31, 2026", fiscal_period: "March 31, 2026", pages: 5, chunks: 20 },
];

describe("App", () => {
  it("does not submit when Enter confirms an IME composition", () => {
    const spy = vi.spyOn(api, "askQuestion").mockResolvedValue({ answer: "a", sources: [], refused: false });
    render(<App />);
    const box = screen.getByPlaceholderText(/ask a question/i);
    fireEvent.change(box, { target: { value: "日本語" } });
    fireEvent.keyDown(box, { key: "Enter", isComposing: true });
    expect(spy).not.toHaveBeenCalled();
    fireEvent.keyDown(box, { key: "Enter" });
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("shows the answer after asking", async () => {
    vi.spyOn(api, "askQuestion").mockResolvedValue({ answer: "It is 1.15%.", sources: [], refused: false });
    render(<App />);
    fireEvent.change(screen.getByPlaceholderText(/ask a question/i), { target: { value: "npa?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByText("It is 1.15%.")).toBeInTheDocument();
  });

  it("shows an error alert when the request fails", async () => {
    vi.spyOn(api, "askQuestion").mockRejectedValue(new Error("Boom"));
    render(<App />);
    fireEvent.change(screen.getByPlaceholderText(/ask a question/i), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Boom"));
  });

  it("offers a document picker and sends the chosen doc with the question", async () => {
    vi.spyOn(api, "listDocuments").mockResolvedValue(DOCS);
    const ask = vi.spyOn(api, "askQuestion").mockResolvedValue({ answer: "a", sources: [], refused: false });
    render(<App />);
    const picker = await screen.findByLabelText(/search in/i);
    expect(picker).toHaveValue("");                       // default: all documents
    expect(screen.getByRole("option", { name: /all documents/i })).toBeInTheDocument();
    fireEvent.change(picker, { target: { value: "data_4.pdf" } });
    fireEvent.change(screen.getByPlaceholderText(/ask a question/i), { target: { value: "npa?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(ask).toHaveBeenCalled());
    expect(ask.mock.calls[0][0]).toBe("npa?");
    expect(ask.mock.calls[0][1]).toMatchObject({ doc: "data_4.pdf" });
  });

  it("hides the picker when no documents are listed", async () => {
    vi.spyOn(api, "listDocuments").mockResolvedValue([]);
    render(<App />);
    await waitFor(() => expect(api.listDocuments).toHaveBeenCalled());
    expect(screen.queryByLabelText(/search in/i)).not.toBeInTheDocument();
  });

  it("shows no banner when the backend is ready", async () => {
    render(<App />);
    await waitFor(() => expect(api.getHealth).toHaveBeenCalled());
    expect(screen.queryByTestId("setup-banner")).not.toBeInTheDocument();
  });

  it("explains setup problems reported by /health", async () => {
    api.getHealth.mockResolvedValue({
      ready: false,
      problems: ["The search index isn't built. Run `python ingest.py` in backend/.", "No GROQ_API_KEY is configured."],
    });
    render(<App />);
    const banner = await screen.findByTestId("setup-banner");
    expect(banner).toHaveTextContent(/ingest\.py/);
    expect(banner).toHaveTextContent(/GROQ_API_KEY/);
  });

  it("tells the user how to start the backend when it is unreachable", async () => {
    api.getHealth.mockResolvedValue(null);
    render(<App />);
    expect(await screen.findByTestId("setup-banner")).toHaveTextContent(/uvicorn/);
  });

  it("lets the user cancel a slow request without showing an error", async () => {
    vi.spyOn(api, "askQuestion").mockImplementation(
      (q, { signal }) =>
        new Promise((_, reject) => {
          signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        })
    );
    render(<App />);
    fireEvent.change(screen.getByPlaceholderText(/ask a question/i), { target: { value: "slow?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    fireEvent.click(await screen.findByRole("button", { name: /cancel/i }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Ask" })).toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText(/retrieving passages/i)).not.toBeInTheDocument();
  });
});
