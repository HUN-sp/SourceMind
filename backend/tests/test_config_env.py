"""A malformed numeric env var should fail with a message naming the variable."""
import importlib

import pytest


@pytest.fixture
def reload_config(monkeypatch):
    import dotenv
    import rag.config as cfg_mod

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    yield lambda: importlib.reload(cfg_mod)
    monkeypatch.undo()
    importlib.reload(cfg_mod)


@pytest.mark.parametrize("var,bad", [("TOP_K", "abc"), ("MIN_SIMILARITY", "high"), ("RATE_LIMIT_PER_MINUTE", "1.5.2")])
def test_bad_numeric_env_names_the_variable(monkeypatch, reload_config, var, bad):
    monkeypatch.setenv(var, bad)
    with pytest.raises(ValueError, match=var):
        reload_config()


def test_numeric_env_values_are_applied(monkeypatch, reload_config):
    monkeypatch.setenv("TOP_K", " 12 ")
    monkeypatch.setenv("MIN_SIMILARITY", "0.5")
    cfg = reload_config().Config()
    assert cfg.top_k == 12 and cfg.min_similarity == 0.5
