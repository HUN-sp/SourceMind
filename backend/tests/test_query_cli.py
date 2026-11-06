"""query.py must follow the configured defaults (same as the API) unless a flag
overrides them. Regression: it used to force the cross-encoder ON by default."""
import pytest

from query import build_parser, search_options


def _opts(*argv):
    return search_options(build_parser().parse_args(["a question", *argv]))


def test_defaults_follow_config_not_force_reranker():
    o = _opts()
    assert o["rerank"] is None      # None => search() uses CONFIG.use_reranker
    assert o["hybrid"] is None      # None => CONFIG.use_hybrid
    assert o["top_k"] == 5


def test_flags_override():
    assert _opts("--rerank")["rerank"] is True
    assert _opts("--no-rerank")["rerank"] is False
    assert _opts("--no-hybrid")["hybrid"] is False
    assert _opts("--k", "3")["top_k"] == 3


def test_rerank_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["q", "--rerank", "--no-rerank"])
