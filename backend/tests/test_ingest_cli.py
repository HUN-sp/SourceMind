"""ingest.py end to end on a one-page synthetic PDF (fake models, temp index)."""
import pytest

import ingest
import rag.store as store
from test_extract_robustness import TEXT, _pdf_bytes


def test_ingest_builds_a_searchable_index(tmp_path, no_index, capsys):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "good.pdf").write_bytes(_pdf_bytes(TEXT))
    assert ingest.run(tmp_path / "data", stats_only=False) == 0
    assert store.collection_size() == 1
    assert store.search("gross npa")[0].doc == "good.pdf"
    assert "indexed 1 chunks" in capsys.readouterr().out


def test_stats_only_does_not_write_an_index(tmp_path, no_index):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "good.pdf").write_bytes(_pdf_bytes(TEXT))
    assert ingest.run(tmp_path / "data", stats_only=True) == 0
    assert store.collection_size() == 0


def test_unreadable_corpus_exits_nonzero_with_a_message(tmp_path, no_index, capsys):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "bad.pdf").write_bytes(b"junk")
    assert ingest.run(tmp_path / "data", stats_only=False) == 1
    assert "ERROR" in capsys.readouterr().err


def test_missing_data_dir_exits_nonzero(tmp_path, no_index, capsys):
    assert ingest.run(tmp_path / "nope", stats_only=False) == 1
    assert "ERROR" in capsys.readouterr().err
