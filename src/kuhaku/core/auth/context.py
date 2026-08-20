"""Generic authentication context.

``AuthContext`` is the one identity primitive every layer of kuhaku accepts
(``RAGEngine``, ``Retriever``, audit logging). kuhaku itself never interprets
``roles``/``permissions`` -- only a caller-supplied ``AuthorizationPolicy`` (see
``policy.py``) does. There is no built-in role hierarchy or clearance vocabulary: what a
role or permission string *means* is entirely up to the application wiring this up.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuthContext:
    """The authenticated (or anonymous) identity behind one request.

    Immutable and provider-agnostic: an ``AuthProvider`` (see ``providers.py``) produces
    one from whatever credential it validates (API key, JWT, ...), and a caller-supplied
    ``AuthorizationPolicy`` is the only thing that ever interprets ``roles``/
    ``permissions``.
    """

    identity: str
    is_authenticated: bool = False
    roles: tuple[str, ...] = field(default_factory=tuple)
    permissions: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def anonymous(cls) -> AuthContext:
        """An unauthenticated context: no identity, no roles, no permissions.

        What ``NoOpAuthProvider`` always returns, and what every other built-in
        ``AuthProvider`` falls back to when its credential doesn't validate.
        """

        return cls(identity="", is_authenticated=False, roles=(), permissions=(), metadata={})
