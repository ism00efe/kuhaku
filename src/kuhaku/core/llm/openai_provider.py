"""OpenAI (or any OpenAI-compatible endpoint) via the Chat Completions HTTP API.

Uses ``requests`` for symmetry with the other providers. Because it targets the standard
``/v1/chat/completions`` shape, it also works with compatible servers (vLLM, LM Studio,
etc.) by changing ``OPENAI_BASE_URL``. Selected with ``LLM_PROVIDER=openai``.
"""

from __future__ import annotations

import logging

import requests
from tenacity import retry_if_exception, wait_exponential

from ..retry import CircuitBreaker, CircuitOpenError, call_with_retry
from .base import LLMError, TokenUsage, is_retryable_request_exception

logger = logging.getLogger(__name__)


class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 1024,
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
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai.")
        if not model or not model.strip():
            raise LLMError("OPENAI_MODEL is not set but LLM_PROVIDER=openai.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
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
                name="llm:openai",
            )
            if circuit_breaker_enabled
            else None
        )
        # Set by generate() when OpenAI reports usage; read by TokenTrackingLLM.
        self.last_usage: TokenUsage | None = None

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def generate(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        def _call() -> dict:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,  # type: ignore[arg-type]
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()

        def _call_with_retry() -> dict:
            return call_with_retry(
                _call,
                service="llm",
                enabled=self._retry_enabled,
                max_attempts=self._retry_max_attempts,
                wait=wait_exponential(
                    multiplier=self._retry_backoff_base_seconds,
                    max=self._retry_backoff_max_seconds,
                ),
                retry=retry_if_exception(is_retryable_request_exception),
            )

        try:
            data = (
                self._circuit_breaker.call(_call_with_retry)
                if self._circuit_breaker is not None
                else _call_with_retry()
            )
        except CircuitOpenError as exc:
            raise LLMError(str(exc)) from exc
        except requests.RequestException as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response: {data}") from exc

        # Best-effort token usage logging. Never changes the return type.
        usage = data.get("usage") or {}
        if usage:
            self.last_usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
            logger.info(
                "llm usage",
                extra={
                    "status": "ok",
                    "token_count": usage.get("completion_tokens"),
                    "prompt_token_count": usage.get("prompt_tokens"),
                },
            )
        return content
