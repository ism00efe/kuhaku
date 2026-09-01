"""Shared exception types.

``KuhakuError`` is the single root for every error kuhaku raises deliberately. The
three-tier default-behaviour policy (``kuhaku.core.policy``) hangs off it
(``SecurityComponentError``, ``CustomComponentError``), as does the capability-resolution
layer (``kuhaku.core.resolve``: ``CapabilityUnavailable``, ``ConsentRequired``,
``StoreConflict``, ``ConfigError``). Adding the base is backward compatible -- an
``except SecurityComponentError`` written before this still catches the same errors.

Every capability-resolution message states three things: what was attempted, what is
missing, and the single next step.

``FallbackWarning`` is a ``Warning``, deliberately outside this hierarchy: a fallback that
succeeded is not an error.
"""

from __future__ import annotations


class KuhakuError(Exception):
    """Root of every error kuhaku raises on purpose."""


class SecurityComponentError(KuhakuError):
    """A security-critical component configured as the kuhaku default failed to
    initialize or failed its startup self-check.

    Raised before the application starts serving requests -- never caught internally.
    """


class CustomComponentError(KuhakuError):
    """A user-provided custom component does not satisfy the protocol required at its
    injection point (missing method(s)), or failed validation.

    Raised immediately at injection time -- never caught internally, so a broken custom
    component always fails fast rather than degrading silently.
    """


class CapabilityUnavailable(KuhakuError):
    """A required capability has no usable candidate and no consent to obtain one.

    Also raised when ``KUHAKU_AUTO`` is disabled and the documented baseline for a
    decision is itself unusable -- with auto off there is no fallback.
    """


class ConsentRequired(KuhakuError):
    """Something that was explicitly requested needs an unapproved package install or
    model download, or an approved install cannot run because the interpreter is not
    isolated."""


class StoreConflict(KuhakuError):
    """A second process was detected writing the same vector store."""


class ConfigError(KuhakuError):
    """Configuration is contradictory or invalid."""


class FallbackWarning(UserWarning):
    """Issued when a performance/helper component configured as the kuhaku default
    could not be loaded and kuhaku fell back to a simpler alternative."""
