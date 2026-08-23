"""LLM provider package: interface + factory.

``build_llm_provider(settings)`` is the single place that maps ``LLM_PROVIDER`` to a
concrete implementation. Adding a provider = one new module + one entry here.
"""

from __future__ import annotations

import logging

from ..config import Settings
from .base import LLMError, LLMProvider

logger = logging.getLogger(__name__)


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate the LLM provider selected by ``settings.llm_provider``."""

    provider = settings.llm_provider.strip().lower()

    # Retry/timeout/circuit-breaker kwargs are identical across all four providers
    # (extended to circuit breakers): they are structurally symmetric dependencies (an
    # external LLM API reachable over HTTP), so resilience applies symmetrically --
    # swapping LLM_PROVIDER must never silently drop retry, timeout, or circuit-breaker
    # protection.
    _resilience_kwargs = dict(
        retry_enabled=settings.retry_enabled,
        retry_max_attempts=settings.retry_llm_max_attempts,
        retry_backoff_base_seconds=settings.retry_llm_backoff_base_seconds,
        retry_backoff_max_seconds=settings.retry_llm_backoff_max_seconds,
        circuit_breaker_enabled=settings.circuit_breaker_enabled,
        circuit_breaker_failure_threshold=settings.circuit_breaker_failure_threshold,
        circuit_breaker_reset_timeout_seconds=settings.circuit_breaker_reset_timeout_seconds,
        circuit_breaker_success_threshold=settings.circuit_breaker_success_threshold,
    )

    if provider == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            **_resilience_kwargs,
        )

    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key or "",
            model=settings.anthropic_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            **_resilience_kwargs,
        )

    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key or "",
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            **_resilience_kwargs,
        )

    if provider == "vertex":
        from .vertex_provider import VertexAIProvider

        return VertexAIProvider(
            model=settings.vertex_model,
            project=settings.vertex_project,
            location=settings.vertex_location,
            timeout=settings.llm_timeout_seconds,
            **_resilience_kwargs,
        )

    raise LLMError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
        "Expected one of: ollama, anthropic, openai, vertex."
    )


__all__ = ["LLMProvider", "LLMError", "build_llm_provider"]
