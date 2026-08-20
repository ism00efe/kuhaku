"""Pluggable authentication.

An ``AuthProvider`` validates one kind of credential and produces the ``AuthContext`` it
represents. Three ship here: ``NoOpAuthProvider`` (always anonymous -- the default),
``APIKeyAuthProvider`` (a caller-configured key table), and ``JWTAuthProvider`` (a
locally-verified HS256 JWT, standard library only -- no external JWT dependency).
``AuthProviderRegistry`` is a simple named registry for resolving/swapping providers at
runtime.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Protocol, runtime_checkable

from .context import AuthContext


@runtime_checkable
class AuthProvider(Protocol):
    """Validates a credential and produces the ``AuthContext`` it represents.

    ``credential`` is deliberately untyped: an API-key provider expects a key string, a
    JWT provider expects a bearer token string, a future provider might expect something
    else entirely -- kuhaku imposes no shape on it. Implementations return
    ``AuthContext.anonymous()`` (never raise) when ``credential`` doesn't validate, so a
    caller can always fall back to unauthenticated behavior rather than handling an
    exception on every request.
    """

    def authenticate(self, credential: Any) -> AuthContext: ...


class NoOpAuthProvider:
    """Always returns an anonymous context, regardless of ``credential``.

    The default provider in a fresh :class:`AuthProviderRegistry` -- with no configured
    authentication, kuhaku behaves exactly as it did before this package existed.
    """

    def authenticate(self, credential: Any = None) -> AuthContext:
        return AuthContext.anonymous()


class APIKeyAuthProvider:
    """Validates a static, caller-configured table of API keys.

    Each key maps to the identity/roles/permissions its holder gets -- defined entirely
    by the caller via :meth:`add_key` (or the ``keys`` constructor argument), never a
    built-in vocabulary.
    """

    def __init__(self, keys: dict[str, dict[str, Any]] | None = None) -> None:
        """``keys`` maps a raw API key string to a mapping accepting the same keyword
        arguments as :meth:`add_key` (``identity``, ``roles``, ``permissions``,
        ``metadata``, all optional)."""

        self._keys: dict[str, AuthContext] = {}
        for key, spec in (keys or {}).items():
            self.add_key(key, **spec)

    def add_key(
        self,
        key: str,
        *,
        identity: str | None = None,
        roles: list[str] | tuple[str, ...] = (),
        permissions: list[str] | tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register ``key``, authenticating to the given identity/roles/permissions.
        ``identity`` defaults to the key string itself when not given."""

        if not key:
            raise ValueError("APIKeyAuthProvider.add_key: key must be a non-empty string")
        self._keys[key] = AuthContext(
            identity=identity if identity is not None else key,
            is_authenticated=True,
            roles=tuple(roles),
            permissions=tuple(permissions),
            metadata=metadata or {},
        )

    def remove_key(self, key: str) -> None:
        """Revoke ``key``. A no-op if it was never registered."""

        self._keys.pop(key, None)

    def authenticate(self, credential: Any) -> AuthContext:
        if isinstance(credential, str):
            context = self._keys.get(credential)
            if context is not None:
                return context
        return AuthContext.anonymous()


class JWTError(ValueError):
    """A JWT could not be parsed or failed verification."""


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


class JWTAuthProvider:
    """Validates a JWT locally using a shared HMAC secret (HS256) -- standard library
    only (``hmac``/``hashlib``/``base64``/``json``), no external JWT dependency.

    Verifies the signature and, when present, the ``exp`` claim; extracts identity/
    roles/permissions from configurable claim names (default ``"sub"``/``"roles"``/
    ``"permissions"``) so it fits whatever claim vocabulary the token issuer already
    uses -- kuhaku imposes no fixed claim schema.
    """

    def __init__(
        self,
        secret: str,
        *,
        algorithm: str = "HS256",
        identity_claim: str = "sub",
        roles_claim: str = "roles",
        permissions_claim: str = "permissions",
    ) -> None:
        if algorithm != "HS256":
            raise ValueError(
                f"JWTAuthProvider: unsupported algorithm '{algorithm}' -- only 'HS256' "
                "is implemented (standard-library HMAC, no external JWT dependency)."
            )
        if not secret:
            raise ValueError("JWTAuthProvider: secret must be a non-empty string")
        self._secret = secret.encode("utf-8")
        self._algorithm = algorithm
        self._identity_claim = identity_claim
        self._roles_claim = roles_claim
        self._permissions_claim = permissions_claim

    def _verify(self, token: str) -> dict[str, Any]:
        try:
            header_b64, payload_b64, signature_b64 = token.split(".")
        except ValueError as exc:
            raise JWTError("malformed token: expected header.payload.signature") from exc

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        try:
            actual_sig = _b64url_decode(signature_b64)
        except Exception as exc:
            raise JWTError("malformed token: invalid base64 signature") from exc
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise JWTError("signature verification failed")

        try:
            header = json.loads(_b64url_decode(header_b64))
            payload = json.loads(_b64url_decode(payload_b64))
        except Exception as exc:
            raise JWTError("malformed token: invalid header/payload encoding") from exc

        if header.get("alg") != self._algorithm:
            raise JWTError(
                f"token alg '{header.get('alg')}' does not match configured "
                f"'{self._algorithm}'"
            )

        exp = payload.get("exp")
        if exp is not None and time.time() >= float(exp):
            raise JWTError("token has expired")

        return payload

    def authenticate(self, credential: Any) -> AuthContext:
        if not isinstance(credential, str) or not credential:
            return AuthContext.anonymous()

        token = credential[7:] if credential.startswith("Bearer ") else credential
        try:
            payload = self._verify(token)
        except JWTError:
            return AuthContext.anonymous()

        identity = str(payload.get(self._identity_claim) or "")
        if not identity:
            return AuthContext.anonymous()

        roles = payload.get(self._roles_claim) or []
        permissions = payload.get(self._permissions_claim) or []
        reserved = {self._identity_claim, self._roles_claim, self._permissions_claim, "exp"}
        metadata = {k: v for k, v in payload.items() if k not in reserved}

        return AuthContext(
            identity=identity,
            is_authenticated=True,
            roles=tuple(roles),
            permissions=tuple(permissions),
            metadata=metadata,
        )


class AuthProviderRegistry:
    """Named :class:`AuthProvider` registry: register/unregister providers, resolve by
    name, authenticate through the current default.

    Seeded with :class:`NoOpAuthProvider` under the name ``"noop"`` -- a fresh registry
    authenticates everything as anonymous until a caller registers something else,
    mirroring ``AuthorizationPolicy``'s own zero-configuration default-allow posture.
    """

    def __init__(self) -> None:
        self._providers: dict[str, AuthProvider] = {"noop": NoOpAuthProvider()}
        self._default_name = "noop"

    def register(self, name: str, provider: AuthProvider, *, make_default: bool = False) -> None:
        """Register ``provider`` under ``name``, replacing any existing provider there.
        Pass ``make_default=True`` to also make it the provider :meth:`authenticate`
        uses when no ``provider`` name is given."""

        if not name:
            raise ValueError("AuthProviderRegistry.register: name must be a non-empty string")
        self._providers[name] = provider
        if make_default:
            self._default_name = name

    def unregister(self, name: str) -> None:
        """Remove the provider registered under ``name``.

        Raises if ``name`` is the current default -- callers must explicitly pick a new
        default (:meth:`set_default`) before removing the one in use, rather than
        silently falling back to ``"noop"``.
        """

        if name == self._default_name:
            raise ValueError(
                f"AuthProviderRegistry.unregister: '{name}' is the current default "
                "provider; call set_default() with a different provider first."
            )
        self._providers.pop(name, None)

    def get(self, name: str) -> AuthProvider:
        """The provider registered under ``name``."""

        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(
                f"AuthProviderRegistry.get: no provider registered under '{name}'"
            ) from None

    def set_default(self, name: str) -> None:
        """Make the provider registered under ``name`` the one :meth:`authenticate` uses
        when no ``provider`` name is given."""

        if name not in self._providers:
            raise KeyError(
                f"AuthProviderRegistry.set_default: no provider registered under '{name}'"
            )
        self._default_name = name

    def authenticate(self, credential: Any, *, provider: str | None = None) -> AuthContext:
        """Authenticate ``credential`` via the named provider, or the current default."""

        return self.get(provider or self._default_name).authenticate(credential)
