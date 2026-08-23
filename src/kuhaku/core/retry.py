"""Retry-with-backoff for transient external-dependency failures.

Ollama model-load delays, momentary GPU memory pressure, ChromaDB lock contention, and
CPU spikes during embedding/reranking are usually transient — retrying after a short wait
often succeeds where failing immediately would surface an error to the user for no
lasting reason. ``call_with_retry`` is the one place this policy lives, used identically
by the LLM providers, the embedder, the vector store, and the reranker rather than
reimplemented per component.

Sits strictly between the call and each component's existing exception handling: on
exhaustion the *original* exception is re-raised unchanged, so callers keep wrapping it
into their own ``LLMError``/``EmbeddingServiceError``/``VectorStoreError`` exactly as
before. A fresh ``tenacity.Retrying`` is constructed per call (no shared/module-level
state), so this is thread-safe without any locking — safe for FastAPI's threadpool.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import TypeVar

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
)
from tenacity.retry import retry_base
from tenacity.wait import wait_base

from .observability.metrics import (
    record_retry_attempt,
    record_retry_failure,
    record_retry_success,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_retry_any_exception: retry_base = retry_if_exception(lambda exc: True)


def call_with_retry(
    fn: Callable[[], T],
    *,
    service: str,
    enabled: bool,
    max_attempts: int,
    wait: wait_base,
    retry: retry_base = _retry_any_exception,
) -> T:
    """Call ``fn()``, retrying per ``wait``/``max_attempts`` on exceptions ``retry`` accepts.

    ``service`` is a fixed, low-cardinality label ("llm" | "embedding" | "vectorstore" |
    "reranker") used for the ``kuhaku_retry_*`` metrics. Only genuine retries (more than one
    attempt made) are counted as a "success" or "failure" -- a first-try success touches
    no counters, and an exception the ``retry`` predicate rejects outright (e.g. an LLM
    4xx) fails after exactly one attempt and is likewise not counted as an exhausted
    retry, since no retry was ever attempted.
    """

    if not enabled or max_attempts <= 1:
        return fn()

    def _before_sleep(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "retrying %s call (attempt %d/%d) after: %s",
            service,
            retry_state.attempt_number,
            max_attempts,
            exc,
        )
        record_retry_attempt(service)

    retrying: Retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait,
        retry=retry,
        before_sleep=_before_sleep,
        reraise=True,
    )
    try:
        result = retrying(fn)
    except Exception:
        if retrying.statistics.get("attempt_number", 1) > 1:
            record_retry_failure(service)
        raise
    if retrying.statistics.get("attempt_number", 1) > 1:
        record_retry_success(service)
    return result


class CircuitState(Enum):
    """The three states a :class:`CircuitBreaker` can be in."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open.

    The wrapped function is never invoked — this is a fast-fail, not a retry outcome.
    """


class CircuitBreaker:
    """Thread-safe circuit breaker: stops calling a failing dependency for a cooldown
    period instead of piling up slow, doomed retries against it.

    Sits outside :func:`call_with_retry` in the LLM provider stack (circuit breaker ->
    retry -> the actual call): the breaker decides *whether* an attempt (including all of
    its retries) is allowed at all; ``call_with_retry`` decides how that one attempt is
    retried once it's allowed through.

    States: ``closed`` (normal — every call passes through) -> ``open`` (fast-fail every
    call without invoking it, once ``failure_threshold`` consecutive failures accumulate)
    -> ``half_open`` (after ``reset_timeout_seconds`` elapses, calls are let through again
    as probes) -> back to ``closed`` once ``success_threshold`` probes succeed, or back to
    ``open`` immediately on a single failed probe.

    A single ``threading.Lock`` guards state transitions; the wrapped function itself is
    always invoked outside the lock, so a slow call never blocks other threads from
    reading/transitioning the breaker's state.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 60.0,
        success_threshold: int = 1,
        name: str = "circuit",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout_seconds = reset_timeout_seconds
        self._success_threshold = success_threshold
        self._name = name

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def _before_call(self) -> None:
        """Raise :class:`CircuitOpenError` if the call must be rejected; otherwise, flip
        an expired ``open`` circuit to ``half_open`` so this call can probe it."""

        with self._lock:
            if self._state is not CircuitState.OPEN:
                return
            assert self._opened_at is not None  # OPEN always sets opened_at
            elapsed = time.monotonic() - self._opened_at
            if elapsed < self._reset_timeout_seconds:
                raise CircuitOpenError(
                    f"Circuit '{self._name}' is open "
                    f"({self._reset_timeout_seconds - elapsed:.1f}s until next probe)."
                )
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0

    def _on_success(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._opened_at = None
            else:
                self._failure_count = 0

    def _on_failure(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                # A failed probe reopens the circuit immediately, regardless of
                # failure_threshold -- one bad probe is proof the dependency is still down.
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._failure_count = 0
                self._success_count = 0
                return

            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "circuit '%s' opened after %d consecutive failures",
                    self._name,
                    self._failure_count,
                )

    def call(self, fn: Callable[[], T]) -> T:
        """Call ``fn()`` through the breaker: raises :class:`CircuitOpenError` without
        calling ``fn`` if the circuit is open, otherwise calls it and records the
        outcome."""

        self._before_call()
        try:
            result = fn()
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorator form: ``@breaker`` wraps ``fn`` so every call goes through
        :meth:`call`."""

        @wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> T:
            return self.call(lambda: fn(*args, **kwargs))

        return wrapper
