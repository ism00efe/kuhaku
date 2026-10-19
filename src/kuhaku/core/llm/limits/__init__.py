"""Client-side rate limiting for any ``LLMProvider``.

:class:`RateLimitedLLM` is a decorator with the provider's own ``generate()``/``name``
shape -- the same "observer, not a protocol change" stance as
:class:`~kuhaku.core.llm.token_tracking.TokenTrackingLLM` -- so the code calling the
provider does not know it is there. It paces calls to a provider account's
requests/min, tokens/min and tokens/day (:class:`RateLimiter`), admitting each call on a
pre-call estimate (:class:`CharTokenEstimator` by default) and reconciling it with the
provider's reported usage, and makes every caller back off together on a 429's
``Retry-After``.
"""

from __future__ import annotations

from .estimate import CharTokenEstimator, TokenEstimate, TokenEstimator
from .rate_limit import (
    Clock,
    Permit,
    RateLimitedLLM,
    RateLimiter,
    RateLimits,
    RateLimitTimeoutError,
    SystemClock,
    account_fingerprint,
    clear_shared_rate_limiters,
    is_rate_limit_error,
    rate_limiter_key,
    retry_after_seconds,
    shared_rate_limiter,
)

__all__ = [
    "CharTokenEstimator",
    "Clock",
    "Permit",
    "RateLimitTimeoutError",
    "RateLimitedLLM",
    "RateLimiter",
    "RateLimits",
    "SystemClock",
    "TokenEstimate",
    "TokenEstimator",
    "account_fingerprint",
    "clear_shared_rate_limiters",
    "is_rate_limit_error",
    "rate_limiter_key",
    "retry_after_seconds",
    "shared_rate_limiter",
]
