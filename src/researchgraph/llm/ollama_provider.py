"""Ollama LLM provider using httpx."""
from __future__ import annotations

from typing import Any
import httpx

from researchgraph.llm.provider import LLMProvider
from researchgraph.logging_setup import get_logger

log = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Client for local Ollama chat completion API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b-instruct", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        log.debug("Sending chat request to Ollama: model=%s url=%s", self.model, url)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
