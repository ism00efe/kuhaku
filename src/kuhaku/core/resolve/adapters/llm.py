"""LLM backend candidates: local Ollama plus the credentialed API providers.

Provider names, SDK imports and construction all live here. The ``auto`` order is
Ollama (safest -- local, free, no data leaves the machine) then, on the monetary/privacy
axis, the credentialed API providers in the historical order openai -> anthropic ->
vertex, with groq appended last (§11).
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlparse

from ..cost import Candidate, Cost
from ..environment import Environment
from ..probes import endpoint_reachable, loopback_daemon_reachable

_LOOPBACK = {"localhost", "127.0.0.1", "::1", "ip6-localhost"}


def _resilience_kwargs(settings) -> dict:
    return dict(
        retry_enabled=settings.retry_enabled,
        retry_max_attempts=settings.retry_llm_max_attempts,
        retry_backoff_base_seconds=settings.retry_llm_backoff_base_seconds,
        retry_backoff_max_seconds=settings.retry_llm_backoff_max_seconds,
        circuit_breaker_enabled=settings.circuit_breaker_enabled,
        circuit_breaker_failure_threshold=settings.circuit_breaker_failure_threshold,
        circuit_breaker_reset_timeout_seconds=settings.circuit_breaker_reset_timeout_seconds,
        circuit_breaker_success_threshold=settings.circuit_breaker_success_threshold,
    )


def _ollama_reachable(url: str) -> bool:
    host = urlparse(url if "://" in url else f"//{url}", scheme="http").hostname or "localhost"
    if host in _LOOPBACK:
        return loopback_daemon_reachable(url)
    return endpoint_reachable(url, timeout=0.5)


def build_ollama(settings):
    from ...llm.ollama_provider import OllamaProvider

    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        **_resilience_kwargs(settings),
    )


def build_anthropic(settings):
    from ...llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(
        api_key=settings.anthropic_api_key or "",
        model=settings.anthropic_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
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
        **_resilience_kwargs(settings),
    )


def build_vertex(settings):
    from ...llm.vertex_provider import VertexAIProvider

    return VertexAIProvider(
        model=settings.vertex_model,
        project=settings.vertex_project,
        location=settings.vertex_location,
        timeout=settings.llm_timeout_seconds,
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
        **_resilience_kwargs(settings),
    )


class OllamaAdapter:
    kind = "llm"
    packages = frozenset()  # Ollama is a server, not a pip package

    def __init__(self, settings) -> None:
        self._s = settings

    def probe(self, env: Environment) -> Sequence[Candidate]:
        s = self._s
        return [Candidate(
            id="ollama",
            kind="llm",
            label=f"Ollama (local, {s.ollama_base_url}, model {s.ollama_model})",
            cost=Cost(note="local; uses this machine's RAM/CPU; no data leaves the machine"),
            ready=_ollama_reachable(s.ollama_base_url),
            safety_rank=0,
            activate=lambda: build_ollama(s),
        )]

    def baseline(self, env: Environment) -> Candidate:
        s = self._s
        return Candidate(
            id="ollama", kind="llm", label="Ollama (local)",
            cost=Cost(note="documented baseline"),
            ready=False, safety_rank=0, activate=lambda: build_ollama(s),
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
    registry.register(OllamaAdapter(settings))
    registry.register(ApiLLMAdapter(settings))
