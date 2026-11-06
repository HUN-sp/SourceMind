"""Ingest must survive one bad PDF and must fail with a clear message (not a raw
traceback) when nothing usable is found."""
import pytest

from rag import extract
from rag.extract import extract_corpus

TEXT = "Gross NPAs to Gross Advances 1.15 percent as at the reporting date for the bank"


def _pdf_bytes(text: str) -> bytes:
    """Smallest valid one-page PDF with a real text layer (pdfminer tolerates the
    approximate xref)."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 700 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_good_pdf_is_extracted(tmp_path):
    (tmp_path / "good.pdf").write_bytes(_pdf_bytes(TEXT))
    pages = extract_corpus(tmp_path)
    assert len(pages) == 1 and "Gross NPAs" in pages[0].text and pages[0].source == "text-layer"


def test_one_corrupt_pdf_is_skipped_not_fatal(tmp_path, caplog):
    (tmp_path / "good.pdf").write_bytes(_pdf_bytes(TEXT))
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf at all")
    with caplog.at_level("WARNING"):
        pages = extract_corpus(tmp_path)
    assert [p.doc for p in pages] == ["good.pdf"]
    assert "broken.pdf" in caplog.text


def test_all_pdfs_unreadable_is_a_clear_error(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"nope")
    with pytest.raises(RuntimeError, match="No text could be extracted"):
        extract_corpus(tmp_path)


def test_no_pdfs_still_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_corpus(tmp_path)
