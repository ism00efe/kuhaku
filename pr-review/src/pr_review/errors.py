"""Exception hierarchy for the review engine."""

from __future__ import annotations


class PRReviewError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(PRReviewError):
    """Configuration is missing or inconsistent."""


class SourceError(PRReviewError):
    """A PR source could not produce a usable :class:`PRContext`."""


class ProviderError(PRReviewError):
    """An LLM provider call failed (network, auth, quota, or bad response)."""


class RateLimited(ProviderError):
    """Provider signalled a retryable throttle (HTTP 429 / 413).

    Transient: the same (provider, model) may succeed after a wait. The caller
    backs off a bounded number of times before moving to the next candidate.
    """

    def __init__(self, message: str, retry_after: float = 60.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderUnavailable(ProviderError):
    """This (provider, model) pair will not work -- retrying is pointless.

    Raised for auth failures and for models the provider does not serve
    (HTTP 400/401/403/404/422). Free catalogues rotate, so a model that
    vanished must fail over immediately rather than burn the retry budget.
    """


class NoProviderAvailable(ProviderError):
    """Every candidate for a tier was unusable (no key, or all exhausted)."""


class RegistryError(PRReviewError):
    """A component was requested by a name that is not registered."""
