import pytest
from pydantic import ValidationError

from researchgraph.config import Settings


def test_password_is_required(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_defaults_and_secret_masking(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "supersecret1")
    s = Settings(_env_file=None)
    assert s.llm_provider == "ollama"
    assert s.raw_dir.name == "raw"
    assert "supersecret1" not in repr(s)
