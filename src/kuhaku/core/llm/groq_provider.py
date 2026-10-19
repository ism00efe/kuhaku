"""Groq via its OpenAI-compatible Chat Completions endpoint.

Groq serves the standard ``/openai/v1/chat/completions`` shape, so this is a thin
subclass of :class:`~kuhaku.core.llm.openai_provider.OpenAIProvider` with Groq's base URL
and its own error text. Selected with ``LLM_PROVIDER=groq``; added to the ``auto`` chain
at the end (see ``kuhaku.core.resolve.adapters.llm``). Free tier as of writing -- prompt
text still leaves the machine. The default model is ``openai/gpt-oss-120b``
(``llama-3.3-70b-versatile`` was shut down for free and developer tiers on 2026-08-16).

Structured output: on the models Groq documents for Structured Outputs
(``openai/gpt-oss-20b``, ``openai/gpt-oss-120b``, ``openai/gpt-oss-safeguard-20b``,
``qwen/qwen3.8-27b`` as of 2026-10) it sends ``response_format={"type":
"json_schema", ...}`` in best-effort mode (``strict: false`` -- strict mode demands every
property be required, which ordinary Pydantic schemas are not). Every other model gets
JSON mode (``{"type": "json_object"}``): valid JSON is guaranteed, the schema itself
travels in the prompt. Groq reports stream usage in ``x_groq.usage`` without being
asked, so ``stream_options`` is not sent.

Images are sent as OpenAI-style ``image_url`` parts; only Groq's vision models accept
them, and Groq takes no PDF input.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .base import LLMError, StructuredOutputMode
from .openai_provider import OpenAIProvider
from .structured import StructuredRequest

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq's vision models (as of 2026-10, Qwen 3.8 27B; the Llama 4 Scout and Maverick
# models were shut down in 2026); every other model it serves is text-only.
_VISION_MARKERS = ("qwen3.8-27b",)

# Models Groq lists as supporting response_format json_schema.
_JSON_SCHEMA_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3.8-27b",
    }
)


class GroqProvider(OpenAIProvider):
    _service = "groq"
    capabilities = replace(
        OpenAIProvider.capabilities, structured_output=StructuredOutputMode.JSON
    )
    _error_prefix = "Groq"
    _stream_usage_option = False
    _encodable_content_types = frozenset({"text", "image"})

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
        if (
            self.capabilities.structured_output is not StructuredOutputMode.NONE
            and self._json_schema_supported()
        ):
            self.capabilities = replace(
                self.capabilities, structured_output=StructuredOutputMode.SCHEMA
            )

    @property
    def name(self) -> str:
        return f"groq:{self._model}"

    def _json_schema_supported(self) -> bool:
        return self._model.strip().lower() in _JSON_SCHEMA_MODELS

    def _default_content_types(self) -> frozenset[str]:
        if any(marker in self._model.lower() for marker in _VISION_MARKERS):
            return frozenset({"text", "image"})
        return frozenset({"text"})

    def _response_format(self, request: StructuredRequest) -> dict[str, Any]:
        if self._json_schema_supported():
            return super()._response_format(request)
        return {"type": "json_object"}
