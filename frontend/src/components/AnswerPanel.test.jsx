import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import AnswerPanel, { formatForCopy } from "./AnswerPanel.jsx";

const RESULT = {
  answer: "Gross NPA ratio was 1.15%.",
  refused: false,
  cached: false,
  sources: [
    { doc: "data_4.pdf", page: 1, snippet: "…", doc_label: "Standalone Results — March 31, 2026" },
    { doc: "data_2.pdf", page: 3, snippet: "…", doc_label: "" },
  ],
};

afterEach(() => vi.unstubAllGlobals());

describe("AnswerPanel", () => {
  it("formats the answer with its citations for copying", () => {
    expect(formatForCopy(RESULT)).toBe(
      "Gross NPA ratio was 1.15%.\n\nSources:\n- Standalone Results — March 31, 2026 (data_4.pdf), page 1\n- data_2.pdf, page 3"
    );
  });

  it("copies to the clipboard and confirms", async () => {
    const writeText = vi.fn().mockResolvedValue();
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    render(<AnswerPanel result={RESULT} />);
    fireEvent.click(screen.getByRole("button", { name: /copy answer/i }));
    await waitFor(() => expect(screen.getByRole("button", { name: /copied/i })).toBeInTheDocument());
    expect(writeText).toHaveBeenCalledWith(formatForCopy(RESULT));
  });

  it("says so when copying is not possible", async () => {
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) } });
    render(<AnswerPanel result={RESULT} />);
    fireEvent.click(screen.getByRole("button", { name: /copy answer/i }));
    expect(await screen.findByRole("button", { name: /copy failed/i })).toBeInTheDocument();
  });

  it("marks cached answers", () => {
    render(<AnswerPanel result={{ ...RESULT, cached: true }} />);
    expect(screen.getByText(/instant answer/i)).toBeInTheDocument();
  });

  it("has no copy button or cached badge on a refusal", () => {
    render(<AnswerPanel result={{ answer: "I couldn't find this.", refused: true, sources: [], cached: false }} />);
    expect(screen.queryByRole("button", { name: /copy/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/instant answer/i)).not.toBeInTheDocument();
  });
});
