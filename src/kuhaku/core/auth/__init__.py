"""Generic authentication and authorization.

Kuhaku-owned, opinion-free about roles, permissions, or clearance levels -- callers
define those entirely themselves, via :class:`~.policy.AuthorizationPolicy`/
:class:`~.providers.AuthProvider` implementations built on the primitives here.

Zero-configuration default: nothing in this package is wired into ``RAGEngine``/
``Retriever`` automatically. An ``auth_context=None`` (the default everywhere it's
accepted) and no configured ``AuthorizationPolicy`` together mean "allow everything" --
identical to kuhaku's behavior before this package existed.
"""

from __future__ import annotations

from .context import AuthContext
from .policy import (
    AuthorizationPolicy,
    ConfigAuthorizationPolicy,
    DefaultAllowPolicy,
    PolicyConfigError,
)
from .providers import (
    APIKeyAuthProvider,
    AuthProvider,
    AuthProviderRegistry,
    JWTAuthProvider,
    JWTError,
    NoOpAuthProvider,
)

__all__ = [
    "AuthContext",
    "AuthorizationPolicy",
    "DefaultAllowPolicy",
    "ConfigAuthorizationPolicy",
    "PolicyConfigError",
    "AuthProvider",
    "AuthProviderRegistry",
    "NoOpAuthProvider",
    "APIKeyAuthProvider",
    "JWTAuthProvider",
    "JWTError",
]
