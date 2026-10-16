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
from .base import LLMError, LLMProvider
from .token_tracking import TokenTrackingLLM

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
        return activate(resolution, env=env, ui=ui)
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
        return provider

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
    return activate(resolution, env=env, ui=ui)


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


__all__ = ["LLMError", "LLMProvider", "build_llm_provider", "with_runtime"]
