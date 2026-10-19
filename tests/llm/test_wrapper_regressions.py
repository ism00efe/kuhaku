"""Regression tests for review findings on the LLM decorators (cache, token
tracking, rate limits, budgets): one section per finding."""

from __future__ import annotations

from types import MappingProxyType

from kuhaku.core.config import Settings
from kuhaku.core.llm import apply_rate_limits
from kuhaku.core.llm.base import LLMResponse, TokenUsage
from kuhaku.core.llm.cache import (
    CachingLLM,
    InMemoryCacheBackend,
    generation_params,
)
from kuhaku.core.llm.limits import (
    RateLimitedLLM,
    RateLimiter,
    RateLimits,
    clear_shared_rate_limiters,
    rate_limiter_key,
)
from kuhaku.core.llm.provider import BaseLLMProvider
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.observability.metrics import LLM_TOKENS_TOTAL
from kuhaku.core.observability.tracing import GEN_AI_SYSTEM, GEN_AI_TOKEN_TYPE
from tests.conftest import prometheus_counter_value as _counter_value
from tests.llm.limits_fakes import FakeClock, FakeProvider


class _Sampled(FakeProvider):
    """A provider that keeps its sampling knobs where the bundled providers do."""

    def __init__(self, temperature: float, max_tokens: int = 512, base_url: str = "http://a"):
        super().__init__(name="openai:gpt-x")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._base_url = base_url


# --- 1. generation params are read through wrappers ------------------------------------


def _rate_limited(provider: FakeProvider) -> RateLimitedLLM:
    return RateLimitedLLM(
        provider, RateLimiter(RateLimits(requests_per_minute=1000), clock=FakeClock())
    )


def test_generation_params_look_through_wrappers():
    expected = {"temperature": 0.7, "max_tokens": 512, "base_url": "http://a"}
    provider = _Sampled(0.7)
    assert generation_params(_rate_limited(provider)) == expected
    assert generation_params(TokenTrackingLLM(provider, provider="openai")) == expected


def test_generation_params_prefer_a_public_mapping():
    class _Explicit(FakeProvider):
        generation_params = MappingProxyType({"temperature": 0.1})

    assert generation_params(_rate_limited(_Explicit())) == {"temperature": 0.1}


def test_cache_behind_a_rate_limiter_misses_on_a_different_temperature():
    backend = InMemoryCacheBackend()
    cold, hot = _Sampled(0.0), _Sampled(1.0)
    CachingLLM(_rate_limited(cold), backend).generate("s", "q")
    CachingLLM(_rate_limited(hot), backend).generate("s", "q")
    assert len(cold.calls) == 1 and len(hot.calls) == 1


# --- 3. token tracking records only this call's usage ----------------------------------


def _tokens(provider: str, kind: str) -> float:
    return _counter_value(LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: provider, GEN_AI_TOKEN_TYPE: kind})


def test_token_tracking_does_not_rebook_stale_usage():
    provider = FakeProvider(script=[("a", TokenUsage(100, 10)), ("b", None)])
    tracker = TokenTrackingLLM(provider, provider="stalecheck")
    before = _tokens("stalecheck", "input")
    tracker.generate("s", "q1")
    tracker.generate("s", "q2")  # reports nothing: last_usage still holds q1's numbers
    assert _tokens("stalecheck", "input") == before + 100


class _ChatOnly(BaseLLMProvider):
    """A v2 provider whose shared ``last_usage`` slot is always wrong: usage must be
    read from the response."""

    def __init__(self) -> None:
        self.last_usage = TokenUsage(999, 999)

    @property
    def name(self) -> str:
        return "chatonly:m"

    def chat(self, messages, *, options=None) -> LLMResponse:
        return LLMResponse(text="ok", usage=TokenUsage(7, 3), model=self.name)


def test_token_tracking_generate_reads_usage_off_the_response():
    tracker = TokenTrackingLLM(_ChatOnly(), provider="respusage")
    before = _tokens("respusage", "input")
    assert tracker.generate("s", "q") == "ok"
    assert _tokens("respusage", "input") == before + 7


# --- 6. shared limiters are per account, not per model name ----------------------------


class _Account(FakeProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1"):
        super().__init__(name="groq:m")
        self._api_key = api_key
        self._base_url = base_url


def test_two_accounts_on_one_model_get_two_limiters():
    clear_shared_rate_limiters()
    try:
        settings = Settings(llm_rate_limit_requests_per_minute=30)
        a = apply_rate_limits(_Account("key-a"), settings)
        a2 = apply_rate_limits(_Account("key-a"), settings)
        b = apply_rate_limits(_Account("key-b"), settings)
        c = apply_rate_limits(_Account("key-a", base_url="http://proxy"), settings)
        assert a.limiter is a2.limiter
        assert b.limiter is not a.limiter
        assert c.limiter is not a.limiter
    finally:
        clear_shared_rate_limiters()


def test_the_limiter_key_never_contains_the_api_key():
    key = rate_limiter_key(_Account("sk-very-secret"))
    assert key.startswith("groq:m@") and "sk-very-secret" not in key
    assert rate_limiter_key(FakeProvider(name="groq:m")) == "groq:m"


