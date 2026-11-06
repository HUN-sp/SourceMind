"""OCR fallback resource handling: open the PDF once per document (not once per
page) and always close it, even when OCR blows up. fitz/RapidOCR are faked so the
test needs neither package."""
import sys
import types

import pytest

from rag import extract
from test_extract_robustness import _pdf_bytes


class _Pix:
    height, width, n = 2, 2, 3
    samples = bytes(12)


class _FitzPage:
    def get_pixmap(self, matrix=None):
        return _Pix()


class _FitzDoc:
    def __init__(self, log):
        self.log = log

    def __getitem__(self, i):
        return _FitzPage()

    def close(self):
        self.log.append("close")


@pytest.fixture
def fakes(monkeypatch):
    log = []
    fitz = types.ModuleType("fitz")
    fitz.open = lambda path: (log.append("open"), _FitzDoc(log))[1]
    fitz.Matrix = lambda a, b: (a, b)
    monkeypatch.setitem(sys.modules, "fitz", fitz)

    state = {"fail": False}

    class RapidOCR:
        def __call__(self, img):
            if state["fail"]:
                raise RuntimeError("ocr exploded")
            return [[None, "OCR TEXT " * 10, 0.9]], None

    ro = types.ModuleType("rapidocr_onnxruntime")
    ro.RapidOCR = RapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", ro)
    monkeypatch.setattr(extract, "_ocr_engine", None)
    return log, state


def test_pdf_is_opened_once_and_closed_once_for_many_ocr_pages(tmp_path, fakes):
    log, _ = fakes
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(_pdf_bytes("tiny", pages=4))  # <50 chars -> every page falls back to OCR
    pages = extract.extract_document(pdf)
    assert [p.source for p in pages] == ["ocr"] * 4
    assert log == ["open", "close"]


def test_pdf_is_closed_when_ocr_raises(tmp_path, fakes):
    log, state = fakes
    state["fail"] = True
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(_pdf_bytes("tiny", pages=2))
    with pytest.raises(RuntimeError, match="ocr exploded"):
        extract.extract_document(pdf)
    assert log == ["open", "close"]


def test_pdf_with_a_text_layer_never_opens_the_ocr_stack(tmp_path, fakes):
    log, _ = fakes
    pdf = tmp_path / "digital.pdf"
    pdf.write_bytes(_pdf_bytes("A long enough text layer line for the reporting period " * 2))
    extract.extract_document(pdf)
    assert log == []
