"""Local LLM via Ollama's HTTP API (the default provider)."""

from __future__ import annotations

import logging

import requests
from tenacity import retry_if_exception, wait_exponential

from ..retry import CircuitBreaker, CircuitOpenError, call_with_retry
from .base import LLMError, TokenUsage, is_retryable_request_exception

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Calls a local Ollama server's ``/api/chat`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 180,
        retry_enabled: bool = True,
        retry_max_attempts: int = 3,
        retry_backoff_base_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 10.0,
        circuit_breaker_enabled: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_reset_timeout_seconds: float = 60.0,
        circuit_breaker_success_threshold: int = 1,
    ) -> None:
        if not base_url or not base_url.strip():
            raise LLMError("OLLAMA_BASE_URL is not set but LLM_PROVIDER=ollama.")
        if not model or not model.strip():
            raise LLMError("OLLAMA_MODEL is not set but LLM_PROVIDER=ollama.")
        self._base_url = base_url.rstrip("/")
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
                name="llm:ollama",
            )
            if circuit_breaker_enabled
            else None
        )
        # Set by generate() when Ollama reports usage; read by TokenTrackingLLM (D33).
        self.last_usage: TokenUsage | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        def _call() -> dict:
            resp = requests.post(
                f"{self._base_url}/api/chat",
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
            raise LLMError(
                f"Failed to reach Ollama at '{self._base_url}' for model '{self._model}'.\n"
                f"Original error: {exc}\n"
                f"Possible solutions:\n"
                f"1. Start Ollama and pull the model:\n"
                f"   ollama serve\n"
                f"   ollama pull {self._model}\n"
                f"2. Or switch to a hosted provider by setting:\n"
                f"   KUHAKU_LLM_PROVIDER=openai and OPENAI_API_KEY=<key>"
            ) from exc
        try:
            content = data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {data}") from exc

        # Best-effort token usage logging (Ollama includes it in the non-streaming
        # response). Never changes the return type — logged only, not surfaced.
        token_count = data.get("eval_count")
        prompt_token_count = data.get("prompt_eval_count")
        if token_count is not None or prompt_token_count is not None:
            self.last_usage = TokenUsage(
                prompt_tokens=prompt_token_count, completion_tokens=token_count
            )
            logger.info(
                "llm usage",
                extra={
                    "status": "ok",
                    "token_count": token_count,
                    "prompt_token_count": prompt_token_count,
                },
            )
        return content
