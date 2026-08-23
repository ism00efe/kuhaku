"""LLM provider interface.

The RAG engine depends only on this protocol — never on a vendor SDK. Concrete providers
live next to this file and are chosen by ``LLM_PROVIDER`` via the factory in
``__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import requests


@dataclass(frozen=True)
class TokenUsage:
    """Token counts from one ``generate()`` call, when the provider's API reports them.

    Not part of the ``LLMProvider`` protocol's return type (``generate()`` still returns
    a plain ``str``): providers instead record their own usage on a
    ``last_usage`` instance attribute, which ``TokenTrackingLLM`` (``llm/token_tracking.py``)
    reads via ``getattr`` after the call completes. This is an addition every provider
    already did the hard part of (all three parse usage out of their response body for
    a log line); it does not change ``generate()``'s signature or return type.
    """

    prompt_tokens: int | None
    completion_tokens: int | None


class LLMError(RuntimeError):
    """Raised when a provider call fails (network, auth, bad response)."""


class LLMUnavailableError(LLMError):
    """The LLM provider could not be reached or timed out for this call.

    Raised by :class:`~kuhaku.tools.rag.engine.RAGEngine` when a provider's
    ``generate()`` raises :class:`LLMError` — translated at that boundary so the API
    layer can map it to a 503 without importing provider internals.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Generates a completion from a system + user prompt."""

    def generate(self, system: str, user: str) -> str: ...

    @property
    def name(self) -> str: ...


def is_retryable_request_exception(exc: BaseException) -> bool:
    """True for connection errors/timeouts and 5xx responses; false for 4xx (retry won't
    help a bad request or auth failure) and anything else.

    Shared by all three providers (Ollama, Anthropic, OpenAI) -- each currently wraps the
    same ``requests.RequestException`` into ``LLMError`` identically, regardless of status
    code, so this is the one place that distinguishes retryable from non-retryable.
    """

    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code >= 500
    return isinstance(exc, requests.RequestException)
