"""Unit tests for LLM provider module."""
from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from researchgraph.config import Settings
from researchgraph.llm import OllamaProvider, OpenAIProvider, get_provider


def test_ollama_provider_complete(monkeypatch):
    def mock_post(self, url, json=None, **kwargs):
        assert url == "http://localhost:11434/api/chat"
        assert json["model"] == "qwen2.5:7b-instruct"
        assert json["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(
            status_code=200,
            json={"message": {"role": "assistant", "content": "Hello!"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    provider = OllamaProvider()
    reply = provider.complete([{"role": "user", "content": "hi"}])
    assert reply == "Hello!"


def test_openai_provider_complete(monkeypatch):
    def mock_post(self, url, headers=None, json=None, **kwargs):
        assert url == "https://api.openai.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "gpt-4o-mini"
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": "OpenAI Response"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    provider = OpenAIProvider(api_key="test-key")
    reply = provider.complete([{"role": "user", "content": "hello"}])
    assert reply == "OpenAI Response"


def test_openai_provider_empty_key():
    with pytest.raises(ValueError):
        OpenAIProvider(api_key="")


def test_get_provider_factory():
    s_ollama = Settings(
        neo4j_password=SecretStr("password123"),
        llm_provider="ollama",
    )
    p_ollama = get_provider(s_ollama)
    assert isinstance(p_ollama, OllamaProvider)

    s_openai = Settings(
        neo4j_password=SecretStr("password123"),
        llm_provider="openai",
        openai_api_key=SecretStr("fake-key"),
    )
    p_openai = get_provider(s_openai)
    assert isinstance(p_openai, OpenAIProvider)
