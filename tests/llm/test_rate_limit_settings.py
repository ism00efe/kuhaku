"""The env-driven ``llm_rate_limit_*`` settings have an effect through
``build_llm_provider`` / ``apply_rate_limits``."""

from __future__ import annotations

import pytest

from kuhaku.core.config import Settings
from kuhaku.core.llm import apply_rate_limits
from kuhaku.core.llm.limits import (
    RateLimitedLLM,
    RateLimits,
    clear_shared_rate_limiters,
)
from tests.llm.limits_fakes import FakeProvider


@pytest.fixture(autouse=True)
def _fresh_shared_limiters():
    clear_shared_rate_limiters()
    yield
    clear_shared_rate_limiters()


def test_unset_leaves_the_provider_unwrapped():
    provider = FakeProvider()
    assert apply_rate_limits(provider, Settings()) is provider


def test_any_limit_wraps_with_a_shared_limiter():
    settings = Settings(llm_rate_limit_tokens_per_minute=8_000, llm_max_tokens=512)
    a = apply_rate_limits(FakeProvider(name="groq:m"), settings)
    b = apply_rate_limits(FakeProvider(name="groq:m"), settings)
    c = apply_rate_limits(FakeProvider(name="groq:other"), settings)
    assert isinstance(a, RateLimitedLLM) and isinstance(b, RateLimitedLLM)
    assert a.limiter is b.limiter
    assert c.limiter is not a.limiter  # Groq meters per model
    assert a.limiter.limits == RateLimits(tokens_per_minute=8_000)


def test_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("KUHAKU_LLM_RATE_LIMIT_REQUESTS_PER_MINUTE", "30")
    monkeypatch.setenv("KUHAKU_LLM_RATE_LIMIT_TOKENS_PER_MINUTE", "8000")
    monkeypatch.setenv("KUHAKU_LLM_RATE_LIMIT_TOKENS_PER_DAY", "200000")
    llm = apply_rate_limits(FakeProvider(name="groq:m"), Settings())
    assert isinstance(llm, RateLimitedLLM)
    assert llm.limiter.limits == RateLimits(
        requests_per_minute=30, tokens_per_minute=8_000, tokens_per_day=200_000
    )


def test_build_llm_provider_applies_the_limits(monkeypatch):
    import kuhaku.core.llm as llm_pkg

    provider = FakeProvider(name="groq:m")
    monkeypatch.setattr(llm_pkg, "activate", lambda resolution, env, ui: provider)
    monkeypatch.setattr(llm_pkg, "resolve", lambda *a, **k: object())

    class _Registry:
        def candidates(self, kind, env):
            return []

    class _Memory:
        def get(self, kind, fp):
            return None

    built = llm_pkg.build_llm_provider(
        Settings(llm_provider="auto", llm_rate_limit_requests_per_minute=10),
        registry=_Registry(),
        env=object(),
        ui=object(),
        memory=_Memory(),
        fingerprint=object(),
    )
    assert isinstance(built, RateLimitedLLM) and built.wrapped is provider
