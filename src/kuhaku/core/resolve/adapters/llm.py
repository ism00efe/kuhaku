"""LLM backend candidates: local Ollama plus the credentialed API providers.

Provider names, SDK imports and construction all live here. A key the user set says
which service they mean to use, so the ``auto`` order is the credentialed API providers
first -- OpenRouter (its default model is free, so no money is spent before a paid
provider is), then openai -> anthropic -> vertex -> groq -- and a local Ollama after
them. Ollama runs a model the user already pulled, never one kuhaku picks: with several,
the resolver asks in a terminal (remembering the answer) and otherwise takes the most
recently pulled one and says so.

:data:`BUILDERS` maps every provider name to its builder, so a pinned provider is built
directly without probing anything -- except a pinned Ollama with no model set, which
needs to know what is pulled (see :func:`kuhaku.core.llm.build_llm_provider`).
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlparse

from ..cost import Candidate, Cost
from ..environment import Environment
from ..probes import endpoint_reachable, loopback_daemon_reachable

_LOOPBACK = {"localhost", "127.0.0.1", "::1", "ip6-localhost"}

# After every API provider (ranks 5-13): a set key takes priority over a local model.
_OLLAMA_RANK = 20


def _resilience_kwargs(settings) -> dict:
    return {
        "retry_enabled": settings.retry_enabled,
        "retry_max_attempts": settings.retry_llm_max_attempts,
        "retry_backoff_base_seconds": settings.retry_llm_backoff_base_seconds,
        "retry_backoff_max_seconds": settings.retry_llm_backoff_max_seconds,
        "circuit_breaker_enabled": settings.circuit_breaker_enabled,
        "circuit_breaker_failure_threshold": settings.circuit_breaker_failure_threshold,
        "circuit_breaker_reset_timeout_seconds": settings.circuit_breaker_reset_timeout_seconds,
        "circuit_breaker_success_threshold": settings.circuit_breaker_success_threshold,
    }


def _configured(settings, field: str):
    """``settings.<field>``, or ``None`` when it still holds the shared default.

    ``llm_temperature``/``llm_max_tokens`` default to values chosen for small local
    models (0.1, 1024). Current Claude models reject a non-default temperature, and
    with adaptive thinking on by default 1024 tokens is often spent before any answer;
    Gemini 3 recommends its own default temperature. So for those providers an
    unchanged default means "let the provider decide". A value equal to the default is
    indistinguishable from it -- set a different one to force it.
    """

    value = getattr(settings, field)
    model_fields = getattr(type(settings), "model_fields", {})
    default = model_fields[field].default if field in model_fields else None
    return None if value == default else value


def _ollama_reachable(url: str) -> bool:
    host = urlparse(url if "://" in url else f"//{url}", scheme="http").hostname or "localhost"
    if host in _LOOPBACK:
        return loopback_daemon_reachable(url)
    return endpoint_reachable(url, timeout=0.5)


def _ollama_models(url: str) -> list[str]:
    """The chat models the Ollama server at ``url`` already has, most recently pulled
    first; ``[]`` when it does not answer. kuhaku uses one of these rather than a model
    of its own choosing: a model that is not pulled fails every call, and pulling one is
    a download nobody asked for. Embedding-only models are left out -- they cannot chat.
    """

    import requests

    try:
        response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=0.5)
        response.raise_for_status()
        entries = [e for e in response.json().get("models", []) if isinstance(e, dict)]
    except (requests.RequestException, ValueError, AttributeError):
        return []
    entries.sort(key=lambda e: str(e.get("modified_at", "")), reverse=True)
    return [str(e["name"]) for e in entries if e.get("name") and not _embedding_only(e)]


def _embedding_only(entry: dict) -> bool:
    details = entry.get("details") or {}
    families = [details.get("family"), *(details.get("families") or [])]
    return "embed" in str(entry["name"]).lower() or any(
        str(family).lower().endswith("bert") for family in families if family
    )


def _ollama_has(models: Sequence[str], model: str) -> bool:
    return model in models or f"{model}:latest" in models


def build_ollama(settings, model: str | None = None):
    from ...llm.base import LLMError
    from ...llm.ollama_provider import OllamaProvider

    model = model or settings.ollama_model
    if not model:
        raise LLMError(
            "No Ollama model is chosen. Set KUHAKU_OLLAMA_MODEL to one you have pulled "
            "(`ollama list` shows them), or leave LLM_PROVIDER on auto to use one of them."
        )
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        content_types=settings.llm_content_types,
        **_resilience_kwargs(settings),
    )


def build_anthropic(settings):
    from ...llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(
        api_key=settings.anthropic_api_key or "",
        model=settings.anthropic_model,
        temperature=_configured(settings, "llm_temperature"),
        max_tokens=_configured(settings, "llm_max_tokens"),
        timeout=settings.llm_timeout_seconds,
        content_types=settings.llm_content_types,
        **_resilience_kwargs(settings),
    )


def build_openai(settings):
    from ...llm.openai_provider import OpenAIProvider

    return OpenAIProvider(
        api_key=settings.openai_api_key or "",
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        content_types=settings.llm_content_types,
        **_resilience_kwargs(settings),
    )


def build_vertex(settings):
    from ...llm.vertex_provider import VertexAIProvider

    return VertexAIProvider(
        model=settings.vertex_model,
        project=settings.vertex_project,
        location=settings.vertex_location,
        temperature=_configured(settings, "llm_temperature"),
        max_tokens=_configured(settings, "llm_max_tokens"),
        timeout=settings.llm_timeout_seconds,
        content_types=settings.llm_content_types,
        **_resilience_kwargs(settings),
    )


def build_groq(settings):
    from ...llm.groq_provider import GroqProvider

    return GroqProvider(
        api_key=settings.groq_api_key or "",
        model=settings.groq_model,
        base_url=settings.groq_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        content_types=settings.llm_content_types,
        **_resilience_kwargs(settings),
    )


def build_openrouter(settings):
    from ...llm.openrouter_provider import OpenRouterProvider

    return OpenRouterProvider(
        api_key=settings.openrouter_api_key or "",
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        **_resilience_kwargs(settings),
    )


BUILDERS = {
    "ollama": build_ollama,
    "openai": build_openai,
    "anthropic": build_anthropic,
    "vertex": build_vertex,
    "groq": build_groq,
    "openrouter": build_openrouter,
}


class OllamaAdapter:
    kind = "llm"
    packages = frozenset()  # Ollama is a server, not a pip package

    def __init__(self, settings) -> None:
        self._s = settings

    def probe(self, env: Environment) -> Sequence[Candidate]:
        """One candidate per pulled model (id ``ollama:<model>``), so the resolver asks
        which one when there are several and remembers the answer; with
        ``ollama_model`` set, that model only (id ``ollama``)."""

        s = self._s
        url = s.ollama_base_url
        models = _ollama_models(url) if _ollama_reachable(url) else None
        local = "local; uses this machine's RAM/CPU; no data leaves the machine"

        if s.ollama_model or not models:
            if models is None:
                ready, note = False, f"no Ollama server answers at {url}"
            elif not models:
                ready, note = False, (
                    "the server is running but has no chat model pulled "
                    "(ollama pull <model>)"
                )
            elif not _ollama_has(models, s.ollama_model):
                ready, note = False, (
                    f"the server is running but model {s.ollama_model} is not pulled "
                    f"(ollama pull {s.ollama_model}, or set KUHAKU_OLLAMA_MODEL to one of "
                    f"{', '.join(models)})"
                )
            else:
                ready, note = True, local
            model_part = f", model {s.ollama_model}" if s.ollama_model else ""
            return [Candidate(
                id="ollama", kind="llm", label=f"Ollama (local, {url}{model_part})",
                cost=Cost(note=note), ready=ready, safety_rank=_OLLAMA_RANK,
                activate=lambda: build_ollama(s),
            )]

        return [
            Candidate(
                id=f"ollama:{model}", kind="llm", label=f"Ollama (local, model {model})",
                cost=Cost(note=local), ready=True, safety_rank=_OLLAMA_RANK,
                activate=(lambda m=model: build_ollama(s, model=m)),
            )
            for model in models
        ]

    def baseline(self, env: Environment) -> Candidate:
        s = self._s
        return Candidate(
            id="ollama", kind="llm", label="Ollama (local)",
            cost=Cost(note="documented baseline"),
            ready=False, safety_rank=_OLLAMA_RANK, activate=lambda: build_ollama(s),
        )


class ApiLLMAdapter:
    kind = "llm"
    packages = frozenset()  # all four talk plain HTTP via requests -- no SDK to install

    def __init__(self, settings) -> None:
        self._s = settings

    def probe(self, env: Environment) -> Sequence[Candidate]:
        s = self._s
        rows = [
            ("openai", bool(s.openai_api_key), 10, True,
             "OpenAI API; billed per use; the prompt is sent to OpenAI", build_openai),
            ("anthropic", bool(s.anthropic_api_key), 11, True,
             "Anthropic API; billed per use; the prompt is sent to Anthropic", build_anthropic),
            ("vertex", bool(s.vertex_project), 12, True,
             "Vertex AI; billed per use; the prompt is sent to Google Cloud", build_vertex),
            ("groq", bool(s.groq_api_key), 13, False,
             "Groq API; free tier; the prompt is sent to Groq", build_groq),
            ("openrouter", bool(s.openrouter_api_key), 5, False,
             "OpenRouter API; free models available; the prompt is sent to OpenRouter",
             build_openrouter),
        ]
        out: list[Candidate] = []
        for cid, has_cred, rank, monetary, note, builder in rows:
            out.append(Candidate(
                id=cid, kind="llm", label=f"{cid} API",
                cost=Cost(network_per_use=True, monetary=monetary, note=note),
                ready=has_cred, safety_rank=rank,
                activate=(lambda b=builder: b(s)),
            ))
        return out

    def baseline(self, env: Environment) -> None:
        return None


def register_llm_adapters(registry, settings) -> None:
    # API providers first: a terminal question lists the options in this order.
    registry.register(ApiLLMAdapter(settings))
    registry.register(OllamaAdapter(settings))
