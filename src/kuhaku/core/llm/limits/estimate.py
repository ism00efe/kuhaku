"""Pre-call token estimates, and reading a provider's post-call usage.

Both limiters have to decide *before* a call whether it may go out, but the real token
count only exists afterwards. So each call is admitted on an estimate and reconciled
against the provider's reported usage once it returns. The estimate only has to be cheap
and roughly right: an over-estimate is credited back on reconciliation, an under-estimate
is debited then -- either way the books end up on the real numbers whenever the provider
reports them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ..base import LLMProvider, TokenUsage


@dataclass(frozen=True)
class TokenEstimate:
    """Expected tokens for one call: prompt (input) and completion (output)."""

    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenEstimator(Protocol):
    """``(system, user) -> TokenEstimate``. Any callable with this shape works -- e.g. a
    wrapper around a real tokenizer when one is installed."""

    def __call__(self, system: str, user: str) -> TokenEstimate: ...


class CharTokenEstimator:
    """Character-count heuristic: ``ceil(chars / chars_per_token) + overhead_tokens``
    input tokens, plus a fixed ``output_tokens`` reservation.

    ``chars_per_token`` defaults to 3.5 rather than the usual English rule of thumb of 4:
    non-English text and JSON tokenize denser, and under-estimating is the expensive
    direction for a rate limiter (it is what produces the 429s this exists to avoid).
    ``output_tokens`` is what to reserve for the completion before it exists; pass the
    provider's own ``max_tokens`` to be strictly safe, or a typical completion length to
    admit more concurrent calls.
    """

    def __init__(
        self,
        *,
        chars_per_token: float = 3.5,
        output_tokens: int = 256,
        overhead_tokens: int = 8,
    ) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive.")
        if output_tokens < 0 or overhead_tokens < 0:
            raise ValueError("output_tokens and overhead_tokens must not be negative.")
        self._chars_per_token = chars_per_token
        self._output_tokens = output_tokens
        self._overhead_tokens = overhead_tokens

    def __call__(self, system: str, user: str) -> TokenEstimate:
        chars = len(system) + len(user)
        return TokenEstimate(
            input_tokens=math.ceil(chars / self._chars_per_token) + self._overhead_tokens,
            output_tokens=self._output_tokens,
        )


def read_usage(provider: LLMProvider) -> TokenUsage | None:
    """The provider's ``last_usage``, duck-typed the same way ``TokenTrackingLLM`` reads
    it -- ``None`` for a provider that does not report usage."""

    return getattr(provider, "last_usage", None)


def fresh_usage(provider: LLMProvider, before: TokenUsage | None) -> TokenUsage | None:
    """Usage the provider reported for the call that just returned, or ``None``.

    Providers only assign ``last_usage`` when a response carries usage and never clear
    it, so after a call that reported nothing the attribute still holds the *previous*
    call's numbers. Each report is a new ``TokenUsage`` instance, so an identity check
    against the value read before the call tells a fresh report from a stale one without
    mutating the provider. (A provider shared across threads can still interleave two
    calls' reports -- ``last_usage`` is a single slot; that limitation predates this.)
    """

    after = read_usage(provider)
    if after is None or after is before:
        return None
    return after


def usage_tokens(usage: TokenUsage) -> tuple[int, int]:
    """``(input, output)`` with unreported halves counted as 0."""

    return usage.prompt_tokens or 0, usage.completion_tokens or 0


def provider_label(provider: LLMProvider) -> str:
    """The ``<provider>`` half of a ``"<provider>:<model>"`` name, used as the metric
    label -- the convention every built-in provider's ``name`` follows."""

    name = provider.name
    return name.split(":", 1)[0] if ":" in name else name


def model_label(provider: LLMProvider) -> str:
    """The ``<model>`` half of a ``"<provider>:<model>"`` name (the whole name if there
    is no prefix)."""

    name = provider.name
    return name.split(":", 1)[1] if ":" in name else name
