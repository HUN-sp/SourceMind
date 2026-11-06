"""Every Python file in backend/ must at least compile.

Guards against a script being corrupted by a bad paste (it happened to
retrieval_harness.py, which had a terminal transcript pasted over its top).
"""
import py_compile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
FILES = sorted(p for p in BACKEND.rglob("*.py") if ".venv" not in p.parts)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(BACKEND)))
def test_python_file_compiles(path):
    py_compile.compile(str(path), doraise=True)
