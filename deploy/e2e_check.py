"""End-to-end check of a deployed SourceMind demo (spends ZERO LLM tokens: only the
pre-computed preset chips are exercised).

    pip install playwright && python -m playwright install chromium   # or use --chrome
    python deploy/e2e_check.py https://<ip-with-dashes>.sslip.io [--chrome PATH] [--shots DIR]

Checks: HTTPS + security headers, /api/health ready, /api/demo presets instant,
each preset answered/refused correctly in <100 ms server time, then in a headless
browser: no login wall, no console/CORS errors, each chip renders its answer or
refusal with citations, and the inspector shows BM25 / BGE / RRF scores.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

EXPECT = {
    "numeric": {"refused": False, "contains": "1.15%", "cites": {("data_4.pdf", 1)}},
    "multihop": {"refused": False, "contains": "1.24%", "cites": {("data_2.pdf", 1), ("data_4.pdf", 1)}},
    "trap": {"refused": True, "contains": "couldn't find", "cites": set()},
}

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, dict(r.headers), r.read()


def post(url: str, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r), (time.perf_counter() - t) * 1000


def http_checks(base: str) -> list[dict]:
    print("HTTP checks")
    status, headers, html = get(base + "/")
    check(status == 200 and b'id="root"' in html, "GET / serves the React app over HTTPS")
    check("max-age" in headers.get("Strict-Transport-Security", ""), "HSTS header present")
    _, _, body = get(base + "/api/health")
    health = json.loads(body)
    check(health.get("ready") is True, f"/api/health ready (chunks={health.get('indexed_chunks')})")
    _, _, body = get(base + "/api/demo")
    presets = json.loads(body)["presets"]
    check([p["id"] for p in presets] == list(EXPECT), "three presets listed")
    check(all(p["instant"] for p in presets), "all presets served from the pre-computed file")
    for p in presets:
        r, rtt = post(base + "/api/ask", {"question": p["question"]})
        exp = EXPECT[p["id"]]
        cites = {(s["doc"], s["page"]) for s in r["sources"]}
        check(r["refused"] == exp["refused"] and exp["contains"] in r["answer"],
              f"{p['id']}: {'refused' if r['refused'] else 'answered'} — {r['answer'][:80]!r}")
        check(cites == exp["cites"], f"{p['id']}: citations {sorted(cites)}")
        check(r["cached"] and r["latency_ms"] < 100, f"{p['id']}: server {r['latency_ms']} ms, round-trip {rtt:.0f} ms, 0 tokens")
        top = r["retrieval"][0]
        check(all(top[k] is not None for k in ("cosine", "bm25", "rrf")), f"{p['id']}: retrieval scores present")
    return presets


def browser_checks(base: str, presets: list[dict], chrome: str | None, shots: str | None) -> None:
    from playwright.sync_api import sync_playwright

    print("Browser checks (headless)")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, executable_path=chrome) if chrome else pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base + "/", wait_until="networkidle")
        check(page.locator("textarea").count() == 1 and page.locator("input[type=password]").count() == 0,
              "page is usable with no login")
        page.wait_for_selector("[data-testid=demo-chips] button")
        for i, p in enumerate(presets):
            page.locator("[data-testid=demo-chips] button").nth(i).click()
            panel = page.locator(".answer-panel")
            panel.wait_for(timeout=15000)
            text = panel.inner_text()
            exp = EXPECT[p["id"]]
            if exp["refused"]:
                check("not found in the documents" in text.lower(), f"{p['id']}: refusal panel shown")
            else:
                check(exp["contains"] in text and "page 1" in text, f"{p['id']}: answer + page citation shown")
            check("pre-computed" in page.locator("[data-testid=response-meta]").inner_text(), f"{p['id']}: marked as pre-computed")
            if i == 0:
                page.get_by_role("button", name="Retrieval & Evals").click()
            lst = page.locator("[data-testid=retrieval-list]")
            lst.wait_for()
            scores = lst.inner_text()
            check(all(k in scores for k in ("BGE cos", "BM25", "RRF")) and lst.locator("li").count() >= 5,
                  f"{p['id']}: inspector shows {lst.locator('li').count()} chunks with BM25/BGE/RRF")
            if shots:
                page.screenshot(path=f"{shots}/{p['id']}.png", full_page=True)
        page.get_by_role("tab", name="Evals & ablation").click()
        check("15/17" in page.locator(".evals").inner_text(), "evals tab shows the reranker ablation")
        if shots:
            page.screenshot(path=f"{shots}/evals.png", full_page=True)
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(base + "/", wait_until="networkidle")
        overflow = mobile.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        check(not overflow, "no horizontal scroll at phone width")
        check(not errors, f"no console/CORS errors {errors[:3] if errors else ''}")
        browser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--chrome", help="path to a Chrome/Chromium binary (skips playwright install)")
    ap.add_argument("--shots", help="directory to save screenshots")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    presets = http_checks(base)
    if not a.no_browser:
        browser_checks(base, presets, a.chrome, a.shots)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
