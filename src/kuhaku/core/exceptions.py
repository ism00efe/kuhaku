"""Exceptions raised by the default behavior policy (``kuhaku.core.policy``).

See DECISIONS.md D53 for the three-tier policy these map to: security-critical
components fail loudly (``SecurityComponentError``), performance/helper components fall
back with a warning (``FallbackWarning``), and user-supplied custom components fail fast
at injection time (``CustomComponentError``).
"""

from __future__ import annotations


class SecurityComponentError(Exception):
    """A security-critical component configured as the kuhaku default failed to
    initialize or failed its startup self-check.

    Raised before the application starts serving requests -- never caught internally.
    """


class CustomComponentError(Exception):
    """A user-provided custom component does not satisfy the protocol required at its
    injection point (missing method(s)), or failed validation.

    Raised immediately at injection time -- never caught internally, so a broken custom
    component always fails fast rather than degrading silently.
    """


class FallbackWarning(UserWarning):
    """Issued when a performance/helper component configured as the kuhaku default
    could not be loaded and kuhaku fell back to a simpler alternative."""
