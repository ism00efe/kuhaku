"""Token-tracking decorator around any :class:`~kuhaku.core.llm.base.LLMProvider`.

An observer, not a protocol change: ``TokenTrackingLLM``
implements the same ``generate()``/``name`` shape and delegates every call to the
wrapped provider, so ``RAGEngine`` neither knows nor cares that usage is being tracked.
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

from ..observability.metrics import record_llm_token_usage
from ..observability.tracing import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    set_current_span_attributes,
)
from .base import LLMProvider

logger = logging.getLogger(__name__)


class TokenTrackingLLM:
    """Wraps an ``LLMProvider``, recording token usage per call."""

    def __init__(self, wrapped: LLMProvider, *, provider: str) -> None:
        self._wrapped = wrapped
        self._provider = provider

    @property
    def name(self) -> str:
        return self._wrapped.name

    def generate(self, system: str, user: str) -> str:
        text = self._wrapped.generate(system, user)
        try:
            self._record_usage()
        except Exception:
            logger.exception("token tracking failed (answer unaffected)")
        return text

    def _record_usage(self) -> None:
        # Duck-typed: only the three concrete providers set this. Any other
        # LLMProvider implementation (e.g. a test fake) simply isn't token-tracked.
        usage = getattr(self._wrapped, "last_usage", None)
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
