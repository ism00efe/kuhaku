"""LLM provider package: interface + factory.

``build_llm_provider(settings)`` maps ``settings.llm_provider`` to a concrete
implementation. ``"auto"`` (the default) goes through the capability resolver
(:mod:`kuhaku.core.resolve`): a reachable local Ollama wins, otherwise the first
credentialed API provider; when several are usable and no terminal is attached the safest
is chosen and announced prominently; when nothing is usable a required build raises
``CapabilityUnavailable``. A concrete ``llm_provider`` value is absolute and is passed to
the resolver as a pin.

Provider construction lives in :mod:`kuhaku.core.resolve.adapters.llm`; adding a provider
is one module there plus a row in ``ApiLLMAdapter``.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..resolve import (
    AUTO,
    ConsoleUI,
    JsonMemory,
    Registry,
    activate,
    probe_environment,
    resolve,
)
from ..resolve.adapters.llm import register_llm_adapters
from .base import LLMError, LLMProvider

logger = logging.getLogger(__name__)

_KNOWN_PROVIDERS = ("ollama", "anthropic", "openai", "vertex", "groq")


def build_llm_provider(
    settings: Settings,
    *,
    env=None,
    ui=None,
    memory=None,
) -> LLMProvider:
    """Instantiate the LLM provider selected by ``settings.llm_provider``.

    ``env`` / ``ui`` / ``memory`` are injection points for tests and for
    ``RAG.__init__`` (which shares one of each across all its decisions); each defaults
    to the real implementation.
    """

    configured = settings.llm_provider.strip().lower()
    if configured != AUTO and configured not in _KNOWN_PROVIDERS:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            f"Expected one of: {', '.join(_KNOWN_PROVIDERS)}."
        )

    registry = Registry()
    register_llm_adapters(registry, settings)
    env = env or probe_environment()
    ui = ui or ConsoleUI()
    memory = memory if memory is not None else JsonMemory(ui=ui)

    resolution = resolve(
        "llm",
        registry=registry,
        env=env,
        ui=ui,
        memory=memory,
        requested=None if configured == AUTO else configured,
        required=True,
    )
    return activate(resolution, env=env, ui=ui)


__all__ = ["LLMError", "LLMProvider", "build_llm_provider"]
