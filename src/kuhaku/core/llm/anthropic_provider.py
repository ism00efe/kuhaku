"""Anthropic Claude via the Messages HTTP API.

Uses ``requests`` (not the vendor SDK) to keep dependencies minimal and the three
providers symmetric. Selected with ``LLM_PROVIDER=anthropic``.
"""

from __future__ import annotations

import logging

import requests
from tenacity import retry_if_exception, wait_exponential

from ..retry import CircuitBreaker, CircuitOpenError, call_with_retry
from .base import LLMError, TokenUsage, is_retryable_request_exception

logger = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class AnthropicProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
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
            raise LLMError("ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic.")
        if not model or not model.strip():
            raise LLMError("ANTHROPIC_MODEL is not set but LLM_PROVIDER=anthropic.")
        self._api_key = api_key
        self._model = model
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
                name="llm:anthropic",
            )
            if circuit_breaker_enabled
            else None
        )
        # Set by generate() when Anthropic reports usage; read by TokenTrackingLLM.
        self.last_usage: TokenUsage | None = None

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def generate(self, system: str, user: str) -> str:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        def _call() -> dict:
            resp = requests.post(
                _API_URL,
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
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        try:
            content = "".join(
                block["text"] for block in data["content"] if block.get("type") == "text"
            ).strip()
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Anthropic response: {data}") from exc

        # Best-effort token usage logging. Never changes the return type.
        usage = data.get("usage") or {}
        if usage:
            self.last_usage = TokenUsage(
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
            )
            logger.info(
                "llm usage",
                extra={
                    "status": "ok",
                    "token_count": usage.get("output_tokens"),
                    "prompt_token_count": usage.get("input_tokens"),
                },
            )
        return content
