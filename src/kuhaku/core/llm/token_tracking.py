"""Token-tracking decorator around any :class:`~kuhaku.core.llm.base.LLMProvider`.

An observer, not a protocol change: ``TokenTrackingLLM`` is an
:class:`~kuhaku.core.llm.wrapper.LLMWrapper`, so it presents the same v1 and v2 surface
as the provider it wraps and delegates every call to it -- ``RAGEngine`` neither knows
nor cares that usage is being tracked.
Recording is strictly best-effort -- a metrics, tracing, or logging failure must never
turn a successful answer into a failed request, so any exception while recording is
caught and logged, never propagated.

Also annotates whatever OTel span is currently open (typically the "generate" span
``RAGEngine`` opens around the call this class wraps -- see
``tools/rag/engine.py``/``core/observability/__init__.py``'s ``instrumented_step``) with
GenAI semantic-convention attributes. This class still knows nothing about span
management itself -- ``set_current_span_attributes`` is a no-op when no span is open.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..models import Message
from ..observability.metrics import record_llm_token_usage
from ..observability.tracing import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    set_current_span_attributes,
)
from .base import (
    GenerationOptions,
    LLMProvider,
    LLMResponse,
    StructuredResponse,
    TokenUsage,
)
from .tool_calling import ToolSpec
from .wrapper import LLMWrapper, generate_response

logger = logging.getLogger(__name__)


class TokenTrackingLLM(LLMWrapper):
    """Wraps an ``LLMProvider``, recording token usage per call.

    Every call records the usage of *that* call. ``generate``/``agenerate`` go through
    :func:`~kuhaku.core.llm.wrapper.generate_response` -- the wrapped provider's
    ``chat`` when it has one, so usage comes off the response; a v1 provider's
    ``last_usage`` only when the call replaced it, so a call that reported nothing is
    not recorded at the previous call's numbers. The v2 methods read usage off the
    response they return -- for a stream, once it has been read to the end -- so
    concurrent calls on one provider are each recorded with their own counts.

    ``generate(s, u)`` and ``agenerate(s, u)`` reach the wrapped provider as the same
    logical call, so a cache underneath keys them identically.
    """

    def __init__(self, wrapped: LLMProvider, *, provider: str) -> None:
        super().__init__(wrapped)
        self._provider = provider

    def generate(self, system: str, user: str) -> str:
        response = generate_response(self._wrapped, system, user)
        self._record(response.usage)
        return response.text

    def chat(
        self, messages: Sequence[Message], *, options: GenerationOptions | None = None
    ) -> LLMResponse:
        response = super().chat(messages, options=options)
        self._record(response.usage)
        return response

    def generate_structured(
        self, messages: Sequence[Message], json_schema: dict[str, Any], **kwargs: Any
    ) -> StructuredResponse:
        response = super().generate_structured(messages, json_schema, **kwargs)
        self._record(response.usage)
        return response

    def chat_with_tools(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], **kwargs: Any
    ) -> LLMResponse:
        response = super().chat_with_tools(messages, tools, **kwargs)
        self._record(response.usage)
        return response

    def _record(self, usage: TokenUsage | None) -> None:
        try:
            self._record_usage(usage)
        except Exception:
            logger.exception("token tracking failed (answer unaffected)")

    def _record_usage(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return

        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        model = self._wrapped.name.removeprefix(f"{self._provider}:")

        record_llm_token_usage(
            provider=self._provider,
            model=model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        set_current_span_attributes(
            **{
                GEN_AI_SYSTEM: self._provider,
                GEN_AI_REQUEST_MODEL: model,
                GEN_AI_USAGE_INPUT_TOKENS: prompt_tokens,
                GEN_AI_USAGE_OUTPUT_TOKENS: completion_tokens,
            }
        )

        logger.info(
            "llm usage",
            extra={
                "provider": self._provider,
                "model": self._wrapped.name,
                "prompt_token_count": prompt_tokens,
                "token_count": completion_tokens,
            },
        )
