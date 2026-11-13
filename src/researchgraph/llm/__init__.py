"""LLM provider abstraction module."""
from __future__ import annotations

from typing import Optional

from researchgraph.config import Settings, get_settings
from researchgraph.llm.ollama_provider import OllamaProvider
from researchgraph.llm.openai_provider import OpenAIProvider
from researchgraph.llm.provider import LLMProvider

__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "get_provider",
]

_DEFAULT_PROVIDER: LLMProvider | None = None


def get_provider(settings: Optional[Settings] = None) -> LLMProvider:
    """Factory to get the configured LLM provider instance."""
    global _DEFAULT_PROVIDER
    if settings is None:
        if _DEFAULT_PROVIDER is None:
            _DEFAULT_PROVIDER = _create_provider(get_settings())
        return _DEFAULT_PROVIDER
    return _create_provider(settings)


def _create_provider(s: Settings) -> LLMProvider:
    if s.llm_provider == "openai":
        key = s.openai_api_key.get_secret_value() if s.openai_api_key else ""
        return OpenAIProvider(
            api_key=key,
            model=s.openai_model,
        )
    return OllamaProvider(
        base_url=s.ollama_base_url,
        model=s.ollama_model,
    )
