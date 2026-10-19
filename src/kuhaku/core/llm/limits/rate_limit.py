"""Client-side rate limiting for LLM providers: requests/min, tokens/min, tokens/day.

Hosted providers enforce per-account quotas and answer an over-quota request with HTTP
429. Discovering the quota that way is expensive: every rejected request still costs a
round trip, the provider's own retry burns its backoff on it, and concurrent callers keep
hammering the same exhausted quota. On Groq's free tier an evaluation of ~500 calls took
hours, almost all of it spent in 429s and retries. A :class:`RateLimiter` paces calls to
the quota *before* they are sent instead.

One limiter per provider account, shared by every wrapper and thread that spends that
account's quota -- the limiter is the unit of sharing, :class:`RateLimitedLLM` is just a
view onto it. Each limit is a token bucket (capacity = the limit, refilled continuously
over its period), so short bursts up to the limit go straight through and sustained load
settles at the limit's rate. The day bucket refills continuously too, which approximates
a provider's fixed daily window; it also starts full on every process start, because the
limiter cannot know what an earlier process already spent.

When a 429 does get through, the provider's ``Retry-After`` hint is fed to
:meth:`RateLimiter.backoff`, which pauses *every* caller of that limiter -- concurrent
calls back off together rather than each discovering the 429 on its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol, TypeVar

from ...models import Message, TextPart, ToolCallPart, ToolResultPart
from ...observability.metrics import record_llm_rate_limit_wait, record_llm_rate_limited
from ...observability.tracing import set_current_span_attributes
from ..base import (
    GenerationOptions,
    LLMError,
    LLMProvider,
    LLMResponse,
    StructuredResponse,
    TokenUsage,
)
from ..tool_calling import ToolSpec
from ..wrapper import LLMWrapper
from .estimate import (
    CharTokenEstimator,
    TokenEstimator,
    fresh_usage,
    provider_label,
    read_usage,
    usage_tokens,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

RATE_LIMIT_WAIT_SECONDS = "kuhaku.llm.rate_limit.wait_seconds"
RATE_LIMIT_RETRIES = "kuhaku.llm.rate_limit.retries"

_MINUTE = 60.0
_DAY = 86_400.0
# A computed wait below this is float noise from the refill arithmetic; sleeping at least
# this long keeps a caller from spinning on a bucket that is a hair short of full.
_MIN_WAIT_SECONDS = 0.001


class Clock(Protocol):
    """Time source for the limiter. Injected so tests run on a fake clock instead of
    really sleeping."""

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The real clock: ``time.monotonic`` / ``time.sleep``."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class RateLimits:
    """Quota for one provider account. ``None`` means "not limited on this axis"."""

    requests_per_minute: float | None = None
    tokens_per_minute: float | None = None
    tokens_per_day: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("requests_per_minute", "tokens_per_minute", "tokens_per_day"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"RateLimits.{field_name} must be positive or None.")

    @property
    def unlimited(self) -> bool:
        return (
            self.requests_per_minute is None
            and self.tokens_per_minute is None
            and self.tokens_per_day is None
        )


class RateLimitTimeoutError(LLMError):
    """The call would have had to wait longer than the limiter's ``max_wait_seconds``.

    An :class:`~kuhaku.core.llm.base.LLMError`, so every call site that already treats a
    provider failure as "LLM unavailable" handles it without a new ``except`` clause.
    """


@dataclass(frozen=True)
class Permit:
    """Admission for one call: the tokens it was charged up front, and how long it
    waited for them. Hand it back to :meth:`RateLimiter.reconcile` or
    :meth:`RateLimiter.release` once the call's outcome is known."""

    tokens: int
    waited_seconds: float = 0.0


class _Bucket:
    """One token bucket. Not thread-safe on its own -- the limiter's lock guards it."""

    def __init__(self, capacity: float, period: float, now: float) -> None:
        self.capacity = capacity
        self._rate = capacity / period
        self._level = capacity
        self._updated = now

    def refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated)
        self._level = min(self.capacity, self._level + elapsed * self._rate)
        self._updated = now

    def wait_for(self, amount: float) -> float:
        return 0.0 if self._level >= amount else (amount - self._level) / self._rate

    def take(self, amount: float) -> None:
        # May go negative on a reconciliation debit: an under-estimated call then makes
        # the next callers wait for the tokens it really used.
        self._level -= amount

    def give(self, amount: float) -> None:
        self._level = min(self.capacity, self._level + amount)


class RateLimiter:
    """Thread-safe token-bucket limiter for one provider account.

    ``max_wait_seconds`` bounds how long :meth:`acquire` may block (``None``: as long as
    it takes); exceeding it raises :class:`RateLimitTimeoutError` without sleeping.

    A single call estimated above a token bucket's whole capacity could never be
    admitted; it is clamped to the capacity (and logged) rather than rejected, because
    the estimate is a heuristic and the provider is the authority on whether the call
    really is too large.
    """

    def __init__(
        self,
        limits: RateLimits,
        *,
        clock: Clock | None = None,
        max_wait_seconds: float | None = None,
        name: str = "llm",
    ) -> None:
        if max_wait_seconds is not None and max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must not be negative.")
        self._limits = limits
        self._clock: Clock = clock or SystemClock()
        self._max_wait_seconds = max_wait_seconds
        self.name = name

        now = self._clock.monotonic()
        self._lock = threading.Lock()
        self._requests = (
            _Bucket(limits.requests_per_minute, _MINUTE, now)
            if limits.requests_per_minute is not None
            else None
        )
        self._token_buckets = [
            _Bucket(limit, period, now)
            for limit, period in (
                (limits.tokens_per_minute, _MINUTE),
                (limits.tokens_per_day, _DAY),
            )
            if limit is not None
        ]
        self._paused_until = now

    @property
    def limits(self) -> RateLimits:
        return self._limits

    def pause_remaining(self) -> float:
        """Seconds left on a :meth:`backoff` pause (0 when not paused)."""

        with self._lock:
            return max(0.0, self._paused_until - self._clock.monotonic())

    def _clamp(self, tokens: int) -> int:
        tokens = max(0, tokens)
        for bucket in self._token_buckets:
            if tokens > bucket.capacity:
                logger.warning(
                    "rate limiter '%s': call estimated at %d tokens exceeds the bucket "
                    "capacity of %d; charging the capacity instead",
                    self.name,
                    tokens,
                    int(bucket.capacity),
                )
                tokens = int(bucket.capacity)
        return tokens

    def try_acquire(self, tokens: int) -> Permit | float:
        """Admit a call costing ``tokens`` now, or say how long to wait.

        Returns a :class:`Permit` (the call may go out; its tokens and one request slot
        are already charged) or the number of seconds to wait before asking again. Never
        blocks -- this is the seam an async caller awaits on (``asyncio.sleep`` on the
        returned float) while :meth:`acquire` is the blocking form.
        """

        tokens = self._clamp(tokens)
        with self._lock:
            now = self._clock.monotonic()
            if now < self._paused_until:
                return max(_MIN_WAIT_SECONDS, self._paused_until - now)
            wait = 0.0
            if self._requests is not None:
                self._requests.refill(now)
                wait = self._requests.wait_for(1)
            for bucket in self._token_buckets:
                bucket.refill(now)
                wait = max(wait, bucket.wait_for(tokens))
            if wait > 0:
                return max(_MIN_WAIT_SECONDS, wait)
            if self._requests is not None:
                self._requests.take(1)
            for bucket in self._token_buckets:
                bucket.take(tokens)
            return Permit(tokens=tokens)

    def acquire(self, tokens: int = 0) -> Permit:
        """Block until a call costing ``tokens`` may go out, then charge it.

        With no argument it admits one request and charges no tokens -- the shape of
        :class:`kuhaku.core.batch.Limiter`'s token-bucket form, so a ``RateLimiter`` can
        be passed straight to ``BatchRunner(limiter=...)`` to pace item attempts by
        request count. (Don't also hand the same limiter to a ``RateLimitedLLM`` making
        those items' calls: each call would then be counted as two requests.)

        Sleeps outside the lock, so one waiting caller never blocks another from being
        admitted or from reconciling.
        """

        waited = 0.0
        while True:
            result = self.try_acquire(tokens)
            if isinstance(result, Permit):
                return replace(result, waited_seconds=waited)
            if self._max_wait_seconds is not None and waited + result > self._max_wait_seconds:
                raise RateLimitTimeoutError(
                    f"Rate limiter '{self.name}' would need {waited + result:.1f}s to admit "
                    f"this call, more than max_wait_seconds={self._max_wait_seconds}."
                )
            self._clock.sleep(result)
            waited += result

    def reconcile(self, permit: Permit, actual_tokens: int) -> None:
        """Correct the up-front charge to what the call really used."""

        delta = actual_tokens - permit.tokens
        if delta == 0:
            return
        with self._lock:
            now = self._clock.monotonic()
            for bucket in self._token_buckets:
                bucket.refill(now)
                if delta > 0:
                    bucket.take(delta)
                else:
                    bucket.give(-delta)

    def release(self, permit: Permit) -> None:
        """Refund a permit's tokens for a call the provider rejected without serving.

        The request slot is not refunded: a rejected request still counted against the
        provider's request quota.
        """

        with self._lock:
            now = self._clock.monotonic()
            for bucket in self._token_buckets:
                bucket.refill(now)
                bucket.give(permit.tokens)

    def backoff(self, seconds: float) -> None:
        """Pause every caller of this limiter for ``seconds`` (a provider's
        ``Retry-After``). Overlapping pauses keep the later end, never shorten one."""

        if seconds <= 0:
            return
        with self._lock:
            until = self._clock.monotonic() + seconds
            self._paused_until = max(self._paused_until, until)
        logger.warning("rate limiter '%s': backing off all callers for %.1fs", self.name, seconds)


_shared_lock = threading.Lock()
_shared: dict[str, RateLimiter] = {}


def shared_rate_limiter(
    key: str,
    limits: RateLimits,
    *,
    clock: Clock | None = None,
    max_wait_seconds: float | None = None,
) -> RateLimiter:
    """The process-wide limiter for ``key`` (e.g. a provider name), created on first use.

    This is what lets two independently built providers for the same account --
    ``RAG()`` and an ``Extractor`` in one process, say -- share one quota. A later call
    with *different* limits replaces the limiter (and says so): the newer configuration
    wins rather than being silently ignored.
    """

    with _shared_lock:
        existing = _shared.get(key)
        if existing is not None and existing.limits == limits:
            return existing
        if existing is not None:
            logger.warning(
                "rate limiter '%s': limits changed from %s to %s; replacing the shared limiter",
                key,
                existing.limits,
                limits,
            )
        limiter = RateLimiter(limits, clock=clock, max_wait_seconds=max_wait_seconds, name=key)
        _shared[key] = limiter
        return limiter


# Attributes that identify the *account* a provider spends -- read duck-typed through
# any wrappers. The API key is only ever hashed: it never appears in a key, a limiter
# name or a log line.
_ACCOUNT_ATTRIBUTES = ("_api_key", "_base_url", "_project", "_location")


def account_fingerprint(provider: object) -> str | None:
    """A short, non-reversible digest of the account ``provider`` spends -- its API key,
    endpoint, and (for Vertex) project and location -- or ``None`` when none of them
    can be read.

    A provider may instead expose a public ``account_fingerprint`` string. Wrappers are
    looked through via ``wrapped``.
    """

    seen: set[int] = set()
    current: object | None = provider
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        explicit = getattr(current, "account_fingerprint", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        values = [
            (attr, getattr(current, attr, None))
            for attr in _ACCOUNT_ATTRIBUTES
            if isinstance(getattr(current, attr, None), str) and getattr(current, attr)
        ]
        if values:
            digest = hashlib.sha256(
                "\0".join(f"{attr}={value}" for attr, value in values).encode("utf-8")
            )
            return digest.hexdigest()[:12]
        current = getattr(current, "wrapped", None)
    return None


def rate_limiter_key(provider: LLMProvider) -> str:
    """The :func:`shared_rate_limiter` key for ``provider``: its ``name``
    (``"<provider>:<model>"``, since quotas are metered per model), qualified by
    :func:`account_fingerprint` when one is known -- two API keys, or two endpoints
    serving the same model name, are two quotas, not one."""

    fingerprint = account_fingerprint(provider)
    return provider.name if fingerprint is None else f"{provider.name}@{fingerprint}"


def clear_shared_rate_limiters() -> None:
    """Forget every process-wide limiter. For tests and for re-configuring a process."""

    with _shared_lock:
        _shared.clear()


# --- recognising a provider's 429 --------------------------------------------------------


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(exc: BaseException) -> int | None:
    # requests/httpx put the status on `.response.status_code`; google-genai's APIError
    # carries it as `.code`, the Anthropic/OpenAI SDKs as `.status_code`.
    response = getattr(exc, "response", None)
    for value in (
        getattr(response, "status_code", None),
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
    ):
        if isinstance(value, int):
            return value
    return None


def _rate_limit_cause(exc: BaseException) -> BaseException | None:
    for link in _exception_chain(exc):
        if _status_code(link) == 429:
            return link
    return None


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when ``exc`` -- or anything in its ``__cause__``/``__context__`` chain -- is
    an HTTP 429. Providers wrap the transport error in ``LLMError``, so the 429 is
    usually one link down."""

    return _rate_limit_cause(exc) is not None


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _parse_duration(value: str) -> float | None:
    # OpenAI/Groq `x-ratelimit-reset-*` format: "7.66s", "1m30s", "120ms".
    parts = _DURATION_PART.findall(value.strip())
    if not parts or "".join(n + u for n, u in parts) != value.strip():
        return None
    return sum(float(number) * _DURATION_UNITS[unit] for number, unit in parts)


def _parse_retry_after(value: str) -> float | None:
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def retry_after_seconds(exc: BaseException) -> float | None:
    """The provider's back-off hint for a 429 in ``exc``'s chain, in seconds.

    ``Retry-After`` (seconds or an HTTP date) first; failing that, the token-quota reset
    header OpenAI-compatible APIs send (``x-ratelimit-reset-tokens``). The request-quota
    reset header is deliberately not used: on Groq it tracks the *daily* request window,
    and waiting for it would stall a run for hours over a per-minute rejection.
    ``None`` when ``exc`` is not a 429 or carries no usable hint.
    """

    cause = _rate_limit_cause(exc)
    if cause is None:
        return None
    headers: Any = getattr(getattr(cause, "response", None), "headers", None)
    if headers is None or not callable(getattr(headers, "get", None)):
        return None
    # requests' and httpx's header mappings are case-insensitive; the second lookup is
    # for a plain dict.
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    reset_tokens = headers.get("x-ratelimit-reset-tokens")
    if retry_after:
        parsed = _parse_retry_after(str(retry_after))
        if parsed is not None:
            return parsed
    if reset_tokens:
        return _parse_duration(str(reset_tokens))
    return None


def _part_text(part: object) -> str:
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, ToolResultPart):
        return part.content
    if isinstance(part, ToolCallPart):
        return part.call.name + json.dumps(part.call.arguments, ensure_ascii=False)
    return ""


def _estimate(estimator: TokenEstimator, messages: Sequence[Message], extra: str = "") -> int:
    """Estimated tokens of a conversation; ``extra`` (a schema, tool definitions) counts
    as input."""

    system: list[str] = []
    rest: list[str] = []
    for message in messages:
        text = "".join(_part_text(part) for part in message.parts)
        (system if message.role == "system" else rest).append(text)
    return estimator("\n".join(system), "\n".join(rest) + extra).total


def _usage(response: LLMResponse | StructuredResponse) -> TokenUsage | None:
    return response.usage


# --- the LLMProvider decorator -----------------------------------------------------------


class RateLimitedLLM(LLMWrapper):
    """Wraps an ``LLMProvider`` so every call is paced by a shared :class:`RateLimiter`.

    Same ``generate()``/``name`` shape as the wrapped provider, so it composes with
    :class:`~kuhaku.core.llm.token_tracking.TokenTrackingLLM` in any order. Per call: estimate
    the tokens, wait for the limiter, call, then reconcile the charge with the usage the
    provider reported (the estimate stays charged when it reports none).

    On a 429 the provider's ``Retry-After`` (or ``default_backoff_seconds`` when it gave
    none) pauses the whole limiter, the call's tokens are refunded, and the call is
    retried up to ``rate_limit_retries`` times through the limiter. These retries are on
    top of the provider's own ``call_with_retry`` attempts, which do not read
    ``Retry-After`` -- keep the count low.

    Every model-calling method goes through :meth:`_paced` with its own estimate: the
    text the call sends plus, for structured output and tool calling, the schema or tool
    definitions. Images and documents are not counted; the reported usage reconciles
    them. ``generate`` is sent as the equivalent ``chat`` so its usage is read off the
    response, not the provider's shared ``last_usage``.
    """

    def __init__(
        self,
        wrapped: LLMProvider,
        limiter: RateLimiter,
        *,
        estimator: TokenEstimator | None = None,
        rate_limit_retries: int = 1,
        default_backoff_seconds: float = 5.0,
    ) -> None:
        if rate_limit_retries < 0:
            raise ValueError("rate_limit_retries must not be negative.")
        super().__init__(wrapped)
        self._limiter = limiter
        self._estimator: TokenEstimator = estimator or CharTokenEstimator()
        self._rate_limit_retries = rate_limit_retries
        self._default_backoff_seconds = default_backoff_seconds

    @property
    def limiter(self) -> RateLimiter:
        return self._limiter

    @property
    def last_usage(self) -> TokenUsage | None:
        # Passed through so an outer TokenTrackingLLM still sees the usage.
        return read_usage(self._wrapped)

    def generate(self, system: str, user: str) -> str:
        messages = [Message.system(system), Message.user(user)]
        return self._paced(
            self._estimator(system, user).total,
            lambda: LLMWrapper.chat(self, messages),
            _usage,
        ).text

    def chat(
        self, messages: Sequence[Message], *, options: GenerationOptions | None = None
    ) -> LLMResponse:
        return self._paced(
            _estimate(self._estimator, messages),
            lambda: LLMWrapper.chat(self, messages, options=options),
            _usage,
        )

    def generate_structured(
        self,
        messages: Sequence[Message],
        json_schema: dict[str, Any],
        *,
        name: str = "response",
        options: GenerationOptions | None = None,
        max_repair_attempts: int = 1,
    ) -> StructuredResponse:
        return self._paced(
            _estimate(self._estimator, messages, json.dumps(json_schema, ensure_ascii=False)),
            lambda: LLMWrapper.generate_structured(
                self,
                messages,
                json_schema,
                name=name,
                options=options,
                max_repair_attempts=max_repair_attempts,
            ),
            _usage,
        )

    def chat_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        tool_choice: str = "auto",
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        definitions = json.dumps(
            [[t.name, t.description, t.parameters] for t in tools], ensure_ascii=False
        )
        return self._paced(
            _estimate(self._estimator, messages, definitions),
            lambda: LLMWrapper.chat_with_tools(
                self, messages, tools, tool_choice=tool_choice, options=options
            ),
            _usage,
        )

    def _paced(
        self,
        tokens: int,
        call: Callable[[], T],
        usage_of: Callable[[T], TokenUsage | None] | None = None,
    ) -> T:
        """Wait for the limiter, call, reconcile. ``usage_of`` reads the call's usage
        off its result; without it, the provider's fresh ``last_usage`` is used."""

        provider = provider_label(self._wrapped)
        retries = 0
        while True:
            permit = self._limiter.acquire(tokens)
            self._observe_wait(provider, permit, retries)
            before = read_usage(self._wrapped)
            try:
                result = call()
            except Exception as exc:
                if not self._should_retry(provider, exc, permit, retries):
                    raise
                retries += 1
                continue
            self._reconcile(permit, result, usage_of, before)
            return result

    # --- stream metering (see limits/v2.py) --------------------------------------------

    def _should_retry(
        self, provider: str, exc: BaseException, permit: Permit, retries: int
    ) -> bool:
        if not is_rate_limit_error(exc):
            # Unknown whether the provider consumed anything: keep the estimate
            # charged, the conservative choice for a quota.
            return False
        self._on_rate_limited(provider, exc, permit)
        return retries < self._rate_limit_retries

    def _reconcile(
        self,
        permit: Permit,
        result: T,
        usage_of: Callable[[T], TokenUsage | None] | None,
        before: TokenUsage | None,
    ) -> None:
        usage = usage_of(result) if usage_of is not None else fresh_usage(self._wrapped, before)
        if usage is not None:
            self._limiter.reconcile(permit, sum(usage_tokens(usage)))

    def _on_rate_limited(self, provider: str, exc: BaseException, permit: Permit) -> None:
        hint = retry_after_seconds(exc)
        self._limiter.release(permit)
        self._limiter.backoff(hint if hint is not None else self._default_backoff_seconds)
        try:
            record_llm_rate_limited(provider=provider)
        except Exception:
            logger.exception("rate-limit metric failed (call unaffected)")

    def _observe_wait(self, provider: str, permit: Permit, retries: int) -> None:
        # Best-effort, like TokenTrackingLLM: telemetry must never fail the call.
        try:
            record_llm_rate_limit_wait(provider=provider, seconds=permit.waited_seconds)
            set_current_span_attributes(
                **{
                    RATE_LIMIT_WAIT_SECONDS: permit.waited_seconds,
                    RATE_LIMIT_RETRIES: retries,
                }
            )
        except Exception:
            logger.exception("rate-limit telemetry failed (call unaffected)")
