"""OpenRouter via its OpenAI-compatible Chat Completions endpoint.

A thin subclass of :class:`~kuhaku.core.llm.openai_provider.OpenAIProvider` with
OpenRouter's base URL and its own error text, like :mod:`.groq_provider`. Selected with
``LLM_PROVIDER=openrouter``, or by ``auto`` when ``OPENROUTER_API_KEY`` is set.
"""

from __future__ import annotations

from .base import LLMError
from .openai_provider import OpenAIProvider

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    _service = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = _OPENROUTER_BASE_URL,
        **kwargs,
    ) -> None:
        if not api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set but LLM_PROVIDER=openrouter. Set the key "
                "(free models need no payment), or choose another provider with "
                "KUHAKU_LLM_PROVIDER."
            )
        if not model or not model.strip():
            raise LLMError("OPENROUTER_MODEL is not set but LLM_PROVIDER=openrouter.")
        super().__init__(api_key, model, base_url=base_url, **kwargs)

    @property
    def name(self) -> str:
        return f"openrouter:{self._model}"
