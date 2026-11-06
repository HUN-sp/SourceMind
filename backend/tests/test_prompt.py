"""The system prompt must explain the two things the model got wrong in live runs:
which column a value belongs to, and what "Q2 FY2026" means in an April-March year."""
from rag.pipeline import SYSTEM_PROMPT, build_context
from rag.store import Retrieved


def test_prompt_explains_labelled_columns_and_fiscal_quarters():
    assert "Year ended March 31, 2026: " in SYSTEM_PROMPT or "labelled" in SYSTEM_PROMPT.lower()
    assert "Q4" in SYSTEM_PROMPT and "March 31" in SYSTEM_PROMPT and "April" in SYSTEM_PROMPT
    assert "Do NOT use" in SYSTEM_PROMPT or "Do NOT use any outside" in SYSTEM_PROMPT   # grounding rule intact


def test_row_chunks_are_marked_as_labelled_rows_in_the_context():
    row = Retrieved(doc="d.pdf", page=1, source="ocr", similarity=0.9, doc_label="Standalone Results — March 31, 2026",
                    chunk_type="row", chunk_index=3,
                    text="Standalone Results — March 31, 2026\nNet Profit | Year ended March 31, 2026: 74671.29")
    ctx = build_context([row])
    assert "[1] d.pdf, page 1" in ctx and "labelled" in ctx.lower()
