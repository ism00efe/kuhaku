"""Answer a prompt for a logical tier, surviving free-tier reality.

Every LLM call in the engine goes through :class:`Dispatcher`. It owns the two
policies that a single provider cannot own by itself:

* **retry** -- a throttle (HTTP 429) or a transient 5xx is waited out, bounded
  by ``concurrency.max_retries``;
* **failover** -- when a candidate is permanently unusable (no key, model
  retired, prompt too large) or its retries run out, the next candidate from
  :meth:`Config.resolve_chain` is tried, including the weaker models borrowed
  from the degradation ladder.

Free catalogues rotate models and hand out small daily quotas, so "this model
is gone" and "come back in a minute" need opposite responses; conflating them
either burns the retry budget on a 404 or gives up on a quota that would have
cleared. Every failover is recorded in :attr:`Dispatcher.notes` so the report
can say which model actually answered.

A dispatcher is also the only thing that sees *all* of a run's calls, so it is
where the run's memory lives. A review issues one call per axis plus one per
verification; without memory each of them independently rediscovers that the
same model is dead or throttled, paying full retries every time. On a free tier
the scarce resource is requests, so a candidate that has already failed is
benched: permanently for the rest of the run when the failure was permanent,
until its cooldown expires when it was a throttle.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from pr_review.config import Candidate, Config
from pr_review.errors import (
    NoProviderAvailable,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)
from pr_review.providers.base import LLMProvider
from pr_review.providers.selector import ModelSelector

_DROPPED = "dropped for the rest of this run"


def _cooling(seconds: float) -> str:
    return f"benched for {int(seconds)}s"


@dataclass(frozen=True)
class Completion:
    """A model's answer plus which candidate produced it."""

    text: str
    candidate: Candidate

    @property
    def degraded(self) -> bool:
        return self.candidate.degraded


class Dispatcher:
    def __init__(self, config: Config, selector: ModelSelector) -> None:
        self.config = config
        self.selector = selector
        # Insertion-ordered event -> count. A PR runs several tasks on the same
        # tier, so the identical failover is recorded once with a count rather
        # than repeated verbatim for every axis.
        self._events: dict[str, int] = {}
        # (provider, model) -> monotonic deadline before which it is skipped.
        # math.inf means "not coming back during this run".
        self._benched: dict[tuple[str, str], float] = {}

    @property
    def notes(self) -> list[str]:
        """Deduplicated failover / degradation events, in the order they arose."""
        return [
            event if count == 1 else f"{event} (x{count})"
            for event, count in self._events.items()
        ]

    def _note(self, event: str) -> None:
        self._events[event] = self._events.get(event, 0) + 1

    # ------------------------------------------------------------------ #

    def complete(
        self,
        tier: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> Completion:
        """Run ``prompt`` on the first candidate of ``tier`` that answers.

        Raises :class:`NoProviderAvailable` when the chain is empty or every
        candidate failed; callers treat that as "this stage did not run", never
        as a finding.
        """
        full = self.config.resolve_chain(tier)
        if not full:
            raise NoProviderAvailable(
                f"tier {tier!r}: no provider configured with a usable API key"
            )

        chain = [c for c in full if self._ready(c)]
        if not chain:
            # Everything is benched. A throttle must not disable a tier for the
            # whole run, so timed benchings are waived when there is nothing
            # else left; permanent ones stay out.
            chain = [c for c in full if not self._retired(c)]
        if not chain:
            raise NoProviderAvailable(
                f"tier {tier!r}: every candidate is permanently unavailable in this run"
            )

        last: Exception | None = None
        for candidate in chain:
            try:
                provider = self.selector.provider_for(candidate)
            except ProviderError as exc:  # unbuildable transport (e.g. no base_url)
                last = exc
                self._bench(candidate, None)
                self._record(tier, candidate, "unusable", exc, _DROPPED)
                continue
            try:
                text = self._attempt(provider, candidate, prompt, system, temperature)
            except ProviderUnavailable as exc:
                last = exc
                self._bench(candidate, None)
                self._record(tier, candidate, "unavailable", exc, _DROPPED)
                continue
            except RateLimited as exc:
                last = exc
                wait = max(exc.retry_after, self.config.concurrency.retry_wait_seconds)
                self._bench(candidate, wait)
                self._record(tier, candidate, "rate-limited", exc, _cooling(wait))
                continue
            except ProviderError as exc:
                last = exc
                wait = self.config.concurrency.retry_wait_seconds
                self._bench(candidate, wait)
                self._record(tier, candidate, "failed", exc, _cooling(wait))
                continue
            if candidate is not full[0]:
                self._record_served(tier, candidate)
            return Completion(text, candidate)

        raise NoProviderAvailable(
            f"tier {tier!r}: all {len(chain)} candidates failed; last error: {last}"
        )

    # -- run memory ----------------------------------------------------- #

    def _bench(self, candidate: Candidate, seconds: float | None) -> None:
        """Take a candidate out of rotation. ``None`` seconds means permanently."""
        key = (candidate.provider, candidate.model)
        deadline = math.inf if seconds is None else time.monotonic() + seconds
        # Never shorten an existing benching.
        self._benched[key] = max(self._benched.get(key, 0.0), deadline)

    def _ready(self, candidate: Candidate) -> bool:
        until = self._benched.get((candidate.provider, candidate.model))
        return until is None or time.monotonic() >= until

    def _retired(self, candidate: Candidate) -> bool:
        return self._benched.get((candidate.provider, candidate.model)) == math.inf

    # ------------------------------------------------------------------ #

    def _attempt(
        self,
        provider: LLMProvider,
        candidate: Candidate,
        prompt: str,
        system: str | None,
        temperature: float | None,
    ) -> str:
        attempts = max(1, self.config.concurrency.max_retries + 1)
        for i in range(attempts):
            try:
                return provider.complete(
                    prompt,
                    model=candidate.model,
                    max_tokens=candidate.max_tokens,
                    temperature=(
                        candidate.temperature if temperature is None else temperature
                    ),
                    system=system,
                )
            except ProviderUnavailable:
                # Permanent for this (provider, model): waiting changes nothing.
                raise
            except RateLimited as exc:
                if i == attempts - 1:
                    raise
                time.sleep(min(exc.retry_after, self.config.concurrency.retry_wait_seconds))
            except ProviderError:
                if i == attempts - 1:
                    raise
                time.sleep(2 * (i + 1))
        raise NoProviderAvailable(f"{candidate.label()}: retries exhausted")

    def _record(
        self, tier: str, candidate: Candidate, reason: str, exc: Exception, disposition: str
    ) -> None:
        self._note(
            f"{tier}: {candidate.label()} {reason} -> failing over, "
            f"{disposition} ({str(exc)[:140]})"
        )

    def _record_served(self, tier: str, candidate: Candidate) -> None:
        suffix = f" (degraded to the {candidate.tier} tier)" if candidate.degraded else ""
        self._note(f"{tier}: served by {candidate.label()}{suffix}")
