"""Tests for the shared retry-with-backoff helper (D40)."""

from __future__ import annotations

import threading
import time

import pytest
from tenacity import retry_if_exception, wait_fixed

from kuhaku.core.observability.metrics import (
    RETRY_ATTEMPTS,
    RETRY_FAILURES,
    RETRY_SUCCESSES,
)
from kuhaku.core.retry import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    call_with_retry,
)
from tests.conftest import prometheus_counter_value as _counter_value


class _Flaky:
    """Calls `fn()` under test, failing the first `fail_times` calls."""

    def __init__(self, fail_times: int, exc: type[BaseException] = ValueError) -> None:
        self.fail_times = fail_times
        self.exc = exc
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc(f"transient failure #{self.calls}")
        return "ok"


def test_first_try_success_touches_no_retry_metrics():
    before_attempts = _counter_value(RETRY_ATTEMPTS, service="llm")
    before_successes = _counter_value(RETRY_SUCCESSES, service="llm")

    flaky = _Flaky(fail_times=0)
    result = call_with_retry(
        flaky, service="llm", enabled=True, max_attempts=3, wait=wait_fixed(0)
    )

    assert result == "ok"
    assert flaky.calls == 1
    assert _counter_value(RETRY_ATTEMPTS, service="llm") == before_attempts
    assert _counter_value(RETRY_SUCCESSES, service="llm") == before_successes


def test_fails_twice_then_succeeds_counts_two_attempts_one_success():
    before_attempts = _counter_value(RETRY_ATTEMPTS, service="embedding")
    before_successes = _counter_value(RETRY_SUCCESSES, service="embedding")

    flaky = _Flaky(fail_times=2)
    result = call_with_retry(
        flaky, service="embedding", enabled=True, max_attempts=3, wait=wait_fixed(0)
    )

    assert result == "ok"
    assert flaky.calls == 3
    assert _counter_value(RETRY_ATTEMPTS, service="embedding") == before_attempts + 2
    assert _counter_value(RETRY_SUCCESSES, service="embedding") == before_successes + 1


def test_exhausted_retries_raises_original_exception_and_counts_one_failure():
    before_failures = _counter_value(RETRY_FAILURES, service="vectorstore")

    flaky = _Flaky(fail_times=5)  # always fails within max_attempts
    with pytest.raises(ValueError, match="transient failure #3"):
        call_with_retry(
            flaky, service="vectorstore", enabled=True, max_attempts=3, wait=wait_fixed(0)
        )

    assert flaky.calls == 3
    assert _counter_value(RETRY_FAILURES, service="vectorstore") == before_failures + 1


def test_non_retryable_exception_fails_immediately_without_counting_a_failure():
    before_attempts = _counter_value(RETRY_ATTEMPTS, service="reranker")
    before_failures = _counter_value(RETRY_FAILURES, service="reranker")

    flaky = _Flaky(fail_times=5)
    with pytest.raises(ValueError):
        call_with_retry(
            flaky,
            service="reranker",
            enabled=True,
            max_attempts=3,
            wait=wait_fixed(0),
            retry=retry_if_exception(lambda exc: False),
        )

    assert flaky.calls == 1  # never retried
    assert _counter_value(RETRY_ATTEMPTS, service="reranker") == before_attempts
    assert _counter_value(RETRY_FAILURES, service="reranker") == before_failures


def test_retry_disabled_calls_once_and_raises_immediately():
    flaky = _Flaky(fail_times=5)
    with pytest.raises(ValueError, match="transient failure #1"):
        call_with_retry(
            flaky, service="llm", enabled=False, max_attempts=3, wait=wait_fixed(0)
        )
    assert flaky.calls == 1


def test_max_attempts_of_one_is_equivalent_to_disabled():
    flaky = _Flaky(fail_times=5)
    with pytest.raises(ValueError, match="transient failure #1"):
        call_with_retry(
            flaky, service="llm", enabled=True, max_attempts=1, wait=wait_fixed(0)
        )
    assert flaky.calls == 1


def test_constant_wait_is_honored_between_attempts(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: sleeps.append(seconds))

    flaky = _Flaky(fail_times=1)
    call_with_retry(
        flaky, service="vectorstore", enabled=True, max_attempts=2, wait=wait_fixed(0.5)
    )

    assert sleeps == [0.5]


# --- CircuitBreaker -----------------------------------------------------------------
def test_circuit_starts_closed():
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=60.0)
    assert breaker.state is CircuitState.CLOSED
    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.state is CircuitState.CLOSED


def test_circuit_opens_after_failure_threshold():
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=60.0)
    flaky = _Flaky(fail_times=99)

    for _ in range(3):
        with pytest.raises(ValueError):
            breaker.call(flaky)

    assert breaker.state is CircuitState.OPEN


def test_open_circuit_rejects_calls_without_invoking_the_function():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=60.0)
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("down")

    with pytest.raises(ValueError):
        breaker.call(boom)
    assert breaker.state is CircuitState.OPEN

    with pytest.raises(CircuitOpenError):
        breaker.call(boom)
    assert calls["n"] == 1  # the second call never reached the wrapped function


def test_circuit_half_opens_after_reset_timeout_and_closes_on_success():
    breaker = CircuitBreaker(
        failure_threshold=1, reset_timeout_seconds=0.01, success_threshold=1
    )

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("down")))
    assert breaker.state is CircuitState.OPEN

    time.sleep(0.3)
    assert breaker.call(lambda: "recovered") == "recovered"
    assert breaker.state is CircuitState.CLOSED


def test_circuit_requires_success_threshold_probes_to_close():
    breaker = CircuitBreaker(
        failure_threshold=1, reset_timeout_seconds=0.01, success_threshold=2
    )

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("down")))
    time.sleep(0.3)

    assert breaker.call(lambda: "probe 1") == "probe 1"
    assert breaker.state is CircuitState.HALF_OPEN  # one probe isn't enough yet

    assert breaker.call(lambda: "probe 2") == "probe 2"
    assert breaker.state is CircuitState.CLOSED


def test_failed_probe_in_half_open_reopens_circuit():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.01)

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("down")))
    time.sleep(0.3)

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("still down")))
    assert breaker.state is CircuitState.OPEN

    # Reopened -- a call before the (fresh) reset timeout elapses is rejected again.
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "should not run")


def test_circuit_breaker_as_decorator():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=60.0)

    @breaker
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_circuit_breaker_is_thread_safe_under_concurrent_failures():
    breaker = CircuitBreaker(failure_threshold=5, reset_timeout_seconds=60.0)
    errors: list[Exception] = []

    def hammer():
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("down")))
        except Exception as exc:  # noqa: BLE001 - collecting both ValueError/CircuitOpenError
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert breaker.state is CircuitState.OPEN
    assert len(errors) == 20  # every call raised something, none crashed the breaker
