"""Groq via its OpenAI-compatible Chat Completions endpoint.

Groq serves the standard ``/openai/v1/chat/completions`` shape, so this is a thin
subclass of :class:`~kuhaku.core.llm.openai_provider.OpenAIProvider` with Groq's base URL
and its own error text. Selected with ``LLM_PROVIDER=groq``; added to the ``auto`` chain
at the end (see ``kuhaku.core.resolve.adapters.llm``). Free tier as of writing -- prompt
text still leaves the machine.
"""

from __future__ import annotations

from .base import LLMError
from .openai_provider import OpenAIProvider

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(OpenAIProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = _GROQ_BASE_URL,
        **kwargs,
    ) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set but LLM_PROVIDER=groq.")
        if not model or not model.strip():
            raise LLMError("GROQ_MODEL is not set but LLM_PROVIDER=groq.")
        super().__init__(api_key, model, base_url=base_url, **kwargs)
        if self._circuit_breaker is not None:
            self._circuit_breaker.name = "llm:groq"

    @property
    def name(self) -> str:
        return f"groq:{self._model}"
