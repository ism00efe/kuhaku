"""Tests for the client-side rate limiter and RateLimitedLLM (core/llm/limits/)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from kuhaku.core.llm.base import LLMError, TokenUsage
from kuhaku.core.llm.limits import (
    CharTokenEstimator,
    Permit,
    RateLimitedLLM,
    RateLimiter,
    RateLimits,
    RateLimitTimeoutError,
    TokenEstimate,
    clear_shared_rate_limiters,
    is_rate_limit_error,
    retry_after_seconds,
    shared_rate_limiter,
)
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.observability.metrics import (
    LLM_RATE_LIMIT_WAIT,
    LLM_RATE_LIMITED_TOTAL,
    LLM_TOKENS_TOTAL,
)
from kuhaku.core.observability.tracing import GEN_AI_SYSTEM, GEN_AI_TOKEN_TYPE
from tests.conftest import prometheus_counter_value, prometheus_histogram_count
from tests.llm.limits_fakes import (
    FakeClock,
    FakeProvider,
    http_error,
    wrapped_http_error,
)


def _fixed(total: int):
    """Estimator charging exactly ``total`` tokens per call."""

    return lambda system, user: TokenEstimate(input_tokens=total, output_tokens=0)


# --- RateLimits ----------------------------------------------------------------------


def test_rate_limits_reject_non_positive_values():
    with pytest.raises(ValueError):
        RateLimits(tokens_per_minute=0)
    assert RateLimits().unlimited
    assert not RateLimits(requests_per_minute=1).unlimited


# --- buckets -------------------------------------------------------------------------


def test_requests_per_minute_allows_a_burst_then_paces():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(requests_per_minute=3), clock=clock)
    for _ in range(3):
        assert limiter.acquire(0).waited_seconds == 0
    permit = limiter.acquire(0)
    assert permit.waited_seconds == pytest.approx(20.0)  # 60s / 3 requests
    assert clock.sleeps == [pytest.approx(20.0)]


def test_tokens_per_minute_waits_for_the_missing_tokens_only():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(tokens_per_minute=6_000), clock=clock)
    limiter.acquire(5_000)
    assert limiter.try_acquire(2_000) == pytest.approx(10.0)  # 1000 short at 100 tok/s
    assert limiter.acquire(2_000).waited_seconds == pytest.approx(10.0)


def test_try_acquire_never_sleeps_and_charges_only_on_admission():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(tokens_per_minute=600), clock=clock)
    assert isinstance(limiter.try_acquire(600), Permit)
    wait = limiter.try_acquire(60)
    assert wait == pytest.approx(6.0)
    assert limiter.try_acquire(60) == pytest.approx(6.0)  # a refusal charged nothing
    assert clock.sleeps == []


def test_tokens_per_day_bucket():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(tokens_per_day=86_400), clock=clock)  # 1 tok/s
    limiter.acquire(86_400)
    assert limiter.try_acquire(100) == pytest.approx(100.0)


def test_oversized_estimate_is_clamped_to_capacity_not_rejected(caplog):
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(tokens_per_minute=1_000), clock=clock)
    permit = limiter.acquire(5_000)
    assert permit.tokens == 1_000
    assert "exceeds the bucket capacity" in caplog.text


def test_reconcile_debits_an_underestimate_and_credits_an_overestimate():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(tokens_per_minute=600), clock=clock)  # 10 tok/s
    permit = limiter.acquire(100)
    limiter.reconcile(permit, 700)  # really used 700 of 600: bucket now at -100
    assert limiter.try_acquire(100) == pytest.approx(20.0)

    clock.advance(60)  # refill to full
    permit = limiter.acquire(600)
    limiter.reconcile(permit, 100)  # over-estimated by 500: credited back
    assert isinstance(limiter.try_acquire(500), Permit)


def test_release_refunds_tokens_but_not_the_request_slot():
    clock = FakeClock()
    limiter = RateLimiter(
        RateLimits(requests_per_minute=1, tokens_per_minute=600), clock=clock
    )
    permit = limiter.acquire(600)
    limiter.release(permit)
    # Tokens are back, but the one request per minute is spent.
    assert limiter.try_acquire(600) == pytest.approx(60.0)


def test_max_wait_raises_without_sleeping():
    clock = FakeClock()
    limiter = RateLimiter(
        RateLimits(tokens_per_minute=60), clock=clock, max_wait_seconds=5
    )
    limiter.acquire(60)
    with pytest.raises(RateLimitTimeoutError) as info:
        limiter.acquire(30)
    assert isinstance(info.value, LLMError)
    assert clock.sleeps == []


def test_backoff_pauses_every_caller_and_keeps_the_later_end():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(requests_per_minute=1_000), clock=clock)
    limiter.backoff(10)
    limiter.backoff(3)  # shorter pause never shortens the running one
    assert limiter.pause_remaining() == pytest.approx(10)
    assert limiter.acquire(0).waited_seconds == pytest.approx(10)
    assert limiter.pause_remaining() == 0


def test_concurrent_callers_never_exceed_capacity():
    clock = FakeClock()  # frozen: nothing refills while the threads race
    limiter = RateLimiter(RateLimits(requests_per_minute=10), clock=clock)
    admitted: list[object] = []
    lock = threading.Lock()
    barrier = threading.Barrier(40)

    def worker() -> None:
        barrier.wait()
        result = limiter.try_acquire(0)
        with lock:
            admitted.append(result)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(isinstance(r, Permit) for r in admitted) == 10


def test_shared_rate_limiter_reuses_and_replaces():
    clear_shared_rate_limiters()
    try:
        a = shared_rate_limiter("groq:m", RateLimits(tokens_per_minute=100))
        assert shared_rate_limiter("groq:m", RateLimits(tokens_per_minute=100)) is a
        b = shared_rate_limiter("groq:m", RateLimits(tokens_per_minute=200))
        assert b is not a and b.limits.tokens_per_minute == 200
        assert shared_rate_limiter("groq:other", RateLimits(tokens_per_minute=100)) is not b
    finally:
        clear_shared_rate_limiters()


# --- recognising 429s ----------------------------------------------------------------


def test_rate_limit_error_found_through_the_llmerror_chain():
    exc = wrapped_http_error(429, {"Retry-After": "7"})
    assert is_rate_limit_error(exc)
    assert retry_after_seconds(exc) == 7.0


def test_non_429_is_not_a_rate_limit():
    exc = wrapped_http_error(503, {"Retry-After": "7"})
    assert not is_rate_limit_error(exc)
    assert retry_after_seconds(exc) is None
    assert not is_rate_limit_error(LLMError("plain"))


def test_retry_after_as_http_date():
    when = datetime.now(UTC) + timedelta(seconds=30)
    exc = http_error(429, {"Retry-After": format_datetime(when, usegmt=True)})
    assert 25 <= retry_after_seconds(exc) <= 31


def test_falls_back_to_token_reset_header_and_ignores_request_reset():
    exc = http_error(
        429, {"x-ratelimit-reset-tokens": "1m2.5s", "x-ratelimit-reset-requests": "2h"}
    )
    assert retry_after_seconds(exc) == pytest.approx(62.5)
    only_requests = http_error(429, {"x-ratelimit-reset-requests": "2h"})
    assert retry_after_seconds(only_requests) is None


def test_unparseable_hint_is_none():
    assert retry_after_seconds(http_error(429, {"Retry-After": "soon"})) is None
    assert retry_after_seconds(http_error(429)) is None


def test_sdk_style_status_attributes_are_recognised():
    class GenaiError(Exception):
        code = 429

    class SdkError(Exception):
        status_code = 429

    assert is_rate_limit_error(GenaiError())
    assert is_rate_limit_error(SdkError())


# --- RateLimitedLLM ------------------------------------------------------------------


def test_wrapper_delegates_name_text_and_usage():
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
    provider = FakeProvider(script=[("hi", usage)])
    llm = RateLimitedLLM(provider, RateLimiter(RateLimits(), clock=FakeClock()))
    assert llm.generate("s", "u") == "hi"
    assert llm.name == "fake:model-x"
    assert llm.last_usage is usage
    assert llm.wrapped is provider
    assert provider.calls == [("s", "u")]


def test_wrapper_reconciles_with_fresh_usage_only():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(tokens_per_minute=600), clock=clock)
    provider = FakeProvider(
        script=[("a", TokenUsage(prompt_tokens=40, completion_tokens=10)), ("b", None)]
    )
    llm = RateLimitedLLM(provider, limiter, estimator=_fixed(100))

    llm.generate("s", "u")  # charged 100, reconciled to 50
    assert isinstance(limiter.try_acquire(550), Permit)
    clock.advance(60)

    # Second call reports nothing; last_usage still holds the first call's numbers,
    # which must not be mistaken for this call's -- the estimate stays charged.
    llm.generate("s", "u")
    assert limiter.try_acquire(550) == pytest.approx(5.0)


def test_wrapper_backs_off_on_429_and_retries():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(requests_per_minute=100), clock=clock)
    provider = FakeProvider(script=[wrapped_http_error(429, {"Retry-After": "12"}), ("ok", None)])
    llm = RateLimitedLLM(provider, limiter, estimator=_fixed(1))
    before = prometheus_counter_value(LLM_RATE_LIMITED_TOTAL, **{GEN_AI_SYSTEM: "fake"})

    assert llm.generate("s", "u") == "ok"
    assert len(provider.calls) == 2
    assert clock.sleeps == [pytest.approx(12.0)]
    assert prometheus_counter_value(
        LLM_RATE_LIMITED_TOTAL, **{GEN_AI_SYSTEM: "fake"}
    ) == before + 1


def test_backoff_from_one_wrapper_holds_back_another_sharing_the_limiter():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(requests_per_minute=100), clock=clock)
    failing = RateLimitedLLM(
        FakeProvider(script=[wrapped_http_error(429, {"Retry-After": "30"})]),
        limiter,
        rate_limit_retries=0,
    )
    other = RateLimitedLLM(FakeProvider(), limiter)
    with pytest.raises(LLMError):
        failing.generate("s", "u")
    other.generate("s", "u")
    assert clock.sleeps == [pytest.approx(30.0)]


def test_429_without_hint_uses_default_backoff_and_reraises_when_exhausted():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(requests_per_minute=100), clock=clock)
    err = wrapped_http_error(429)
    provider = FakeProvider(script=[err, err])
    llm = RateLimitedLLM(provider, limiter, rate_limit_retries=1, default_backoff_seconds=4)
    with pytest.raises(LLMError) as info:
        llm.generate("s", "u")
    assert info.value is err
    assert len(provider.calls) == 2
    assert limiter.pause_remaining() == pytest.approx(4)


def test_other_errors_are_not_retried_and_keep_their_charge():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(tokens_per_minute=600), clock=clock)
    provider = FakeProvider(script=[wrapped_http_error(500)])
    llm = RateLimitedLLM(provider, limiter, estimator=_fixed(600))
    with pytest.raises(LLMError):
        llm.generate("s", "u")
    assert len(provider.calls) == 1
    assert limiter.try_acquire(60) == pytest.approx(6.0)


def test_wait_is_recorded_as_a_metric():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(requests_per_minute=1), clock=clock)
    llm = RateLimitedLLM(FakeProvider(name="waitprov:m"), limiter)
    before = prometheus_histogram_count(LLM_RATE_LIMIT_WAIT, **{GEN_AI_SYSTEM: "waitprov"})
    llm.generate("s", "u")
    llm.generate("s", "u")
    assert prometheus_histogram_count(
        LLM_RATE_LIMIT_WAIT, **{GEN_AI_SYSTEM: "waitprov"}
    ) == before + 2


def test_char_estimator():
    est = CharTokenEstimator(chars_per_token=4, output_tokens=100, overhead_tokens=0)
    assert est("abcd", "efgh") == TokenEstimate(input_tokens=2, output_tokens=100)
    with pytest.raises(ValueError):
        CharTokenEstimator(chars_per_token=0)


@pytest.mark.parametrize("tracking_outside", [True, False])
def test_composes_with_token_tracking_in_either_order(tracking_outside):
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(tokens_per_minute=600), clock=clock)
    provider = FakeProvider(
        name="composeprov:m",
        script=[("x", TokenUsage(prompt_tokens=30, completion_tokens=20))],
    )
    if tracking_outside:
        llm = TokenTrackingLLM(
            RateLimitedLLM(provider, limiter, estimator=_fixed(600)), provider="composeprov"
        )
    else:
        llm = RateLimitedLLM(
            TokenTrackingLLM(provider, provider="composeprov"), limiter, estimator=_fixed(600)
        )
    before = prometheus_counter_value(
        LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "composeprov", GEN_AI_TOKEN_TYPE: "input"}
    )
    assert llm.generate("s", "u") == "x"
    # Token tracking saw the usage...
    assert prometheus_counter_value(
        LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "composeprov", GEN_AI_TOKEN_TYPE: "input"}
    ) == before + 30
    # ...and the limiter reconciled 600 down to the real 50.
    assert isinstance(limiter.try_acquire(550), Permit)
