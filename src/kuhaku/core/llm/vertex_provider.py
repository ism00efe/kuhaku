"""Google Vertex AI (Gemini) via the ``google-genai`` SDK.

Optional provider for users with Google Cloud credits who want to avoid running a local
model. The SDK is imported lazily so ``google-genai`` stays an optional dependency (see
``pyproject.toml``'s ``vertex`` extra) -- importing this module without it installed does
not fail; only instantiating :class:`VertexAIProvider` does. Selected with
``LLM_PROVIDER=vertex``.
"""

from __future__ import annotations

import logging
import os

from tenacity import retry_if_exception, wait_exponential

from ..retry import CircuitBreaker, CircuitOpenError, call_with_retry
from .base import LLMError, TokenUsage

logger = logging.getLogger(__name__)

_MISSING_CONFIG_MESSAGE = """Vertex AI configuration is missing. Please set `vertex_project` and `vertex_location` via Settings:

from kuhaku.core.config import get_settings
get_settings().vertex_project = "your-project-id"
get_settings().vertex_location = "us-central1\""""


def is_retryable_vertex_exception(exc: BaseException) -> bool:
    """True for connection errors/timeouts and 5xx API errors; false for 4xx (retry won't
    help a bad request or auth failure) and anything else -- mirrors
    ``is_retryable_request_exception`` (D40) for the google-genai SDK's own exception
    types, which are not ``requests.RequestException`` subclasses.
    """

    import httpx
    import requests
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        return exc.code >= 500
    return isinstance(exc, (httpx.RequestError, requests.RequestException))


class VertexAIProvider:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        project: str | None = None,
        location: str | None = None,
        *,
        timeout: int = 120,
        retry_enabled: bool = True,
        retry_max_attempts: int = 3,
        retry_backoff_base_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 10.0,
        circuit_breaker_enabled: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_reset_timeout_seconds: float = 60.0,
        circuit_breaker_success_threshold: int = 1,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Vertex AI support requires: pip install google-genai"
            ) from exc

        self._model = model
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        if not self._project or not self._location:
            raise ValueError(_MISSING_CONFIG_MESSAGE)
        try:
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - SDK exception types must not leak
            raise ValueError(_MISSING_CONFIG_MESSAGE) from exc

        self._timeout = timeout
        self._retry_enabled = retry_enabled
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_base_seconds = retry_backoff_base_seconds
        self._retry_backoff_max_seconds = retry_backoff_max_seconds
        self._circuit_breaker = (
            CircuitBreaker(
                failure_threshold=circuit_breaker_failure_threshold,
                reset_timeout_seconds=circuit_breaker_reset_timeout_seconds,
                success_threshold=circuit_breaker_success_threshold,
                name="llm:vertex",
            )
            if circuit_breaker_enabled
            else None
        )
        # Set by generate() when Vertex reports usage; read by TokenTrackingLLM (D33).
        self.last_usage = None

    @property
    def name(self) -> str:
        return f"vertex:{self._model}"

    def generate(self, system: str, user: str) -> str:
        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=user,
                config={
                    "system_instruction": system,
                    # SDK timeout is in milliseconds.
                    "http_options": {"timeout": self._timeout * 1000},
                },
            )

        def _call_with_retry():
            return call_with_retry(
                _call,
                service="llm",
                enabled=self._retry_enabled,
                max_attempts=self._retry_max_attempts,
                wait=wait_exponential(
                    multiplier=self._retry_backoff_base_seconds,
                    max=self._retry_backoff_max_seconds,
                ),
                retry=retry_if_exception(is_retryable_vertex_exception),
            )

        try:
            response = (
                self._circuit_breaker.call(_call_with_retry)
                if self._circuit_breaker is not None
                else _call_with_retry()
            )
        except CircuitOpenError as exc:
            raise LLMError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - SDK exception types must not leak (D33)
            raise LLMError(f"Vertex AI request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMError(f"Unexpected Vertex AI response: {response!r}")

        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            self.last_usage = TokenUsage(
                prompt_tokens=getattr(usage_metadata, "prompt_token_count", None),
                completion_tokens=getattr(usage_metadata, "candidates_token_count", None),
            )

        return text.strip()
