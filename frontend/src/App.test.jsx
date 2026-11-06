import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App.jsx";
import * as api from "./api.js";

afterEach(() => vi.restoreAllMocks());

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
});
