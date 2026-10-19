"""LLM provider package: interface + factory.

``build_llm_provider(settings)`` maps ``settings.llm_provider`` to a concrete
implementation. ``"auto"`` (the default) goes through the capability resolver
(:mod:`kuhaku.core.resolve`): the provider used last wins and is announced; otherwise a
provider whose key is set -- the free OpenRouter first, then the other credentialed API
providers -- and only without one a local Ollama, running one of the models already
pulled on it; when several are usable a terminal is asked (and the answer remembered),
without one the safest is chosen and announced prominently; when nothing is usable a
required build raises ``CapabilityUnavailable``. Only credentials and the Ollama endpoint are checked -- the
GPU probes import torch and an LLM backend does not need them. A concrete value is built
directly and recorded as the last one used.

:func:`with_runtime` wraps a provider in every layer core applies to model calls. A tool
calls it instead of wrapping by hand, so a layer added here reaches every tool.

Provider construction lives in :mod:`kuhaku.core.resolve.adapters.llm`; adding a provider
is one module there plus a row in ``ApiLLMAdapter``.

Every provider the factory builds implements the v2 surface
(:class:`~kuhaku.core.llm.base.ChatLLMProvider`: ``chat``, ``generate_structured``,
``stream``, their ``a*`` async twins and ``capabilities``) on top of v1's
``generate(system, user)``; :func:`as_chat_provider` lifts any other v1 provider to it.
They also implement tool calling (:class:`ToolCallingLLMProvider`: ``chat_with_tools``
and its stream/async twins, taking :class:`ToolSpec` objects);
:func:`as_tool_calling_provider` lifts anything else, by emulation.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..resolve import (
    AUTO,
    JsonMemory,
    Registry,
    activate,
    default_ui,
    probe_environment,
    resolve,
)
from ..resolve import (
    fingerprint as build_fingerprint,
)
from ..resolve.adapters.llm import BUILDERS, register_llm_adapters
from .base import (
    ChatLLMProvider,
    GenerationOptions,
    LLMCapabilities,
    LLMError,
    LLMProvider,
    LLMResponse,
    StructuredOutputError,
    StructuredOutputMode,
    StructuredResponse,
    TokenUsage,
    ToolCallingLLMProvider,
    UnsupportedContentError,
)
from .cache import (
    CacheBackend,
    CacheEntry,
    CacheStats,
    CachingLLM,
    InMemoryCacheBackend,
    SQLiteCacheBackend,
    llm_cache_key,
)
from .limits import (
    CharTokenEstimator,
    RateLimitedLLM,
    RateLimits,
    rate_limiter_key,
    shared_rate_limiter,
)
from .provider import (
    BaseLLMProvider,
    LegacyLLMAdapter,
    as_chat_provider,
    capabilities_of,
)
from .token_tracking import TokenTrackingLLM
from .tool_calling import (
    ToolRequest,
    ToolSpec,
    execute_tool_call,
    tool_result_message,
)
from .wrapper import LLMWrapper, as_tool_calling_provider

logger = logging.getLogger(__name__)


def build_llm_provider(
    settings: Settings,
    *,
    registry: Registry | None = None,
    env=None,
    ui=None,
    memory=None,
    fingerprint=None,
) -> LLMProvider:
    """Instantiate the LLM provider selected by ``settings.llm_provider``.

    ``registry`` / ``env`` / ``ui`` / ``memory`` / ``fingerprint`` are injection points
    for tests and for ``RAG.__init__`` (which shares one of each across all its
    decisions). ``registry``, when given, must already have the LLM adapters registered;
    otherwise a fresh one is built here.
    """

    if registry is None:
        registry = Registry()
        register_llm_adapters(registry, settings)
    env = env or probe_environment(hardware=False)
    ui = ui or default_ui()
    memory = memory if memory is not None else JsonMemory(ui=ui)

    configured = settings.llm_provider.strip().lower()
    if configured == "ollama" and not settings.ollama_model:
        # Pinned to Ollama without a model: one of the pulled models, chosen exactly as
        # auto chooses among them (asked in a terminal, remembered).
        local = [c for c in registry.candidates("llm", env) if _is_ollama(c.id)]
        resolution = resolve(
            "llm", registry=registry, env=env, ui=ui, memory=memory, required=True,
            candidates=local, fingerprint=fingerprint,
        )
        return apply_rate_limits(activate(resolution, env=env, ui=ui), settings)
    if configured != AUTO:
        # A pinned provider is built as asked, with no probing; it is still recorded so
        # that a later "auto" run continues with the provider used last.
        builder = BUILDERS.get(configured)
        if builder is None:
            raise LLMError(
                f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
                f"Expected one of: {', '.join(sorted([AUTO, *BUILDERS]))}."
            )
        provider = builder(settings)
        fp = fingerprint or build_fingerprint(env, packages=registry.required_packages())
        if memory.get("llm", fp) != configured:
            memory.put("llm", fp, configured)
        return apply_rate_limits(provider, settings)

    candidates = registry.candidates("llm", env)
    fingerprint = fingerprint or build_fingerprint(env, packages=registry.required_packages())
    if any(c.ready and not _is_ollama(c.id) for c in candidates) and not _is_ollama(
        memory.get("llm", fingerprint) or ""
    ):
        # A key the user set wins over a local model -- unless a local model is what
        # they used last, which keeps winning like any provider used last.
        candidates = [c for c in candidates if not _is_ollama(c.id)]
    resolution = resolve(
        "llm",
        registry=registry,
        env=env,
        ui=ui,
        memory=memory,
        required=True,
        candidates=candidates,
        fingerprint=fingerprint,
    )
    return apply_rate_limits(activate(resolution, env=env, ui=ui), settings)


def _is_ollama(candidate_id: str) -> bool:
    return candidate_id == "ollama" or candidate_id.startswith("ollama:")


def with_runtime(llm: LLMProvider, *, token_tracking: bool = True) -> LLMProvider:
    """``llm`` wrapped in every layer core applies to model calls -- today, token
    accounting (:class:`TokenTrackingLLM`). Safe to call twice on the same provider."""

    if token_tracking and not isinstance(llm, TokenTrackingLLM):
        # A provider's name is "<provider>:<model>"; the part before the colon is the
        # label TokenTrackingLLM puts on its metrics.
        llm = TokenTrackingLLM(llm, provider=llm.name.split(":", 1)[0])
    return llm


def apply_rate_limits(provider: LLMProvider, settings: Settings) -> LLMProvider:
    """Wrap ``provider`` in a :class:`~kuhaku.core.llm.limits.RateLimitedLLM` when any
    ``llm_rate_limit_*`` setting is set; return it unchanged otherwise.

    The limiter is the process-wide one for the provider's model *and account*
    (:func:`~kuhaku.core.llm.limits.rate_limiter_key`: its ``name`` plus a hash of its
    API key / endpoint), so every provider built for the same account and model shares
    one quota, and two accounts never share one. The output reservation is
    ``llm_max_tokens`` -- the most a completion can cost -- and is credited back once
    the provider reports what the call really used.
    """

    limits = RateLimits(
        requests_per_minute=settings.llm_rate_limit_requests_per_minute,
        tokens_per_minute=settings.llm_rate_limit_tokens_per_minute,
        tokens_per_day=settings.llm_rate_limit_tokens_per_day,
    )
    if limits.unlimited:
        return provider
    return RateLimitedLLM(
        provider,
        shared_rate_limiter(rate_limiter_key(provider), limits),
        estimator=CharTokenEstimator(output_tokens=settings.llm_max_tokens),
    )


__all__ = [
    "BaseLLMProvider",
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "CachingLLM",
    "ChatLLMProvider",
    "GenerationOptions",
    "InMemoryCacheBackend",
    "LLMCapabilities",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "LLMWrapper",
    "LegacyLLMAdapter",
    "SQLiteCacheBackend",
    "StructuredOutputError",
    "StructuredOutputMode",
    "StructuredResponse",
    "TokenTrackingLLM",
    "TokenUsage",
    "ToolCallingLLMProvider",
    "ToolRequest",
    "ToolSpec",
    "UnsupportedContentError",
    "apply_rate_limits",
    "as_chat_provider",
    "as_tool_calling_provider",
    "build_llm_provider",
    "capabilities_of",
    "execute_tool_call",
    "llm_cache_key",
    "tool_result_message",
    "with_runtime",
]

