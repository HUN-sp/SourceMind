"""requirements-ci.txt (what CI installs) must not drift from the real pins."""
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def _pins(name: str) -> dict[str, str]:
    out = {}
    for line in (BACKEND / name).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s#]+)", line.strip())
        if m:
            out[m.group(1).lower()] = m.group(3)
    return out


def test_ci_pins_match_runtime_and_dev_pins():
    real = {**_pins("requirements.txt"), **_pins("requirements-dev.txt")}
    ci = _pins("requirements-ci.txt")
    assert ci, "requirements-ci.txt has no pins?"
    drift = {k: (v, real.get(k)) for k, v in ci.items() if k in real and real[k] != v}
    assert not drift, f"requirements-ci.txt differs from requirements*.txt: {drift}"
    unknown = set(ci) - set(real)
    assert not unknown, f"pinned in CI but not in requirements*.txt: {unknown}"
