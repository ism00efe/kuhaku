"""LLM provider package: interface + factory.

``build_llm_provider(settings)`` is the single place that maps ``LLM_PROVIDER`` to a
concrete implementation. Adding a provider = one new module + one entry here.
"""

from __future__ import annotations

import logging

from ..config import Settings
from .base import LLMError, LLMProvider

logger = logging.getLogger(__name__)


def _ollama_reachable(base_url: str, *, timeout: float = 1.5) -> bool:
    """Best-effort local Ollama reachability probe, used only to pick a default
    provider -- never called when the user explicitly asked for ``ollama``, so an
    explicit choice always gets the real, informative error from OllamaProvider itself
    on first use."""

    if not base_url or not base_url.strip():
        return False
    try:
        import requests

        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return resp.ok
    except requests.RequestException:
        return False


def _resolve_default_provider(settings: Settings) -> str:
    """No ``LLM_PROVIDER`` was set by the user: probe the local Ollama default and,
    only if it is unreachable, fall back to the first hosted provider that has
    credentials configured -- logging the switch either way."""

    if _ollama_reachable(settings.ollama_base_url):
        return "ollama"

    for name, credential in (
        ("anthropic", settings.anthropic_api_key),
        ("openai", settings.openai_api_key),
        ("vertex", settings.vertex_project),
    ):
        if credential:
            logger.warning(
                "Local Ollama is not reachable at '%s'; no LLM_PROVIDER was set, so "
                "falling back to '%s' (credentials found).",
                settings.ollama_base_url,
                name,
            )
            return name

    raise LLMError(
        "No LLM backend is available: local Ollama is not reachable at "
        f"'{settings.ollama_base_url}', and no hosted provider credentials were found.\n"
        "Pick one of the following:\n"
        "1. To use a hosted provider, set an API key, e.g.:\n"
        "   KUHAKU_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY=sk-...\n"
        "2. To use a local model, install Ollama and pull a model, e.g.:\n"
        "   ollama serve\n"
        f"   ollama pull {settings.ollama_model}"
    )


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Instantiate the LLM provider selected by ``settings.llm_provider``.

    If the user never set ``LLM_PROVIDER`` explicitly (it's still the ``"ollama"``
    default), this probes the local Ollama default first and transparently falls back
    to a hosted provider with configured credentials when it's unreachable -- see
    :func:`_resolve_default_provider`. An explicit ``LLM_PROVIDER=ollama`` is always
    honored as-is; if Ollama is actually unreachable, the existing clear error from
    :class:`~kuhaku.core.llm.ollama_provider.OllamaProvider` surfaces on first use,
    with no silent fallback.
    """

    provider = settings.llm_provider.strip().lower()
    if provider == "ollama" and "llm_provider" not in settings.model_fields_set:
        provider = _resolve_default_provider(settings)

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
