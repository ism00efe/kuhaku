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
import binascii
import hashlib
import hmac
import json
import logging
import math
import re
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..exceptions import KuhakuError
from .context import AuthContext

logger = logging.getLogger(__name__)


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


def _key_digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).digest()


class APIKeyAuthProvider:
    """Validates a static, caller-configured table of API keys.

    Each key maps to the identity/roles/permissions its holder gets -- defined entirely
    by the caller via :meth:`add_key` (or the ``keys`` constructor argument), never a
    built-in vocabulary.

    Keys are held only as SHA-256 digests: a lookup hashes the presented credential,
    finds the entry by digest and confirms it with :func:`hmac.compare_digest`. The raw
    key never becomes an identity -- the identity flows into audit records and session
    owner columns, so a key used as one would write the secret itself to disk.
    """

    def __init__(self, keys: dict[str, dict[str, Any]] | None = None) -> None:
        """``keys`` maps a raw API key string to a mapping accepting the same keyword
        arguments as :meth:`add_key` (``identity``, ``roles``, ``permissions``,
        ``metadata``, all optional)."""

        self._keys: dict[bytes, AuthContext] = {}
        for key, spec in (keys or {}).items():
            self.add_key(key, **spec)

    @staticmethod
    def key_label(key: str) -> str:
        """The default identity for ``key``: ``"key:"`` plus the first 12 hex digits of
        its SHA-256 -- stable per key, but not reversible to it."""

        return "key:" + _key_digest(key).hex()[:12]

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
        ``identity`` defaults to :meth:`key_label` -- never the key itself."""

        if not isinstance(key, str) or not key:
            raise ValueError("APIKeyAuthProvider.add_key: key must be a non-empty string")
        if identity is not None and identity == key:
            raise ValueError(
                "APIKeyAuthProvider.add_key: identity must not be the key itself -- the "
                "identity is written to audit records. Omit it to get a hashed label."
            )
        self._keys[_key_digest(key)] = AuthContext(
            identity=identity if identity is not None else self.key_label(key),
            is_authenticated=True,
            roles=tuple(roles),
            permissions=tuple(permissions),
            metadata=metadata or {},
        )

    def remove_key(self, key: str) -> None:
        """Revoke ``key``. A no-op if it was never registered."""

        if isinstance(key, str):
            self._keys.pop(_key_digest(key), None)

    def authenticate(self, credential: Any) -> AuthContext:
        if not isinstance(credential, str) or not credential:
            return AuthContext.anonymous()
        digest = _key_digest(credential)
        context = self._keys.get(digest)
        # The dict lookup hashes the digest, not the key; the match is then confirmed
        # with compare_digest so no byte-wise early exit ever runs on secret material.
        if context is not None and any(hmac.compare_digest(k, digest) for k in self._keys):
            return context
        return AuthContext.anonymous()


class JWTError(KuhakuError, ValueError):
    """A JWT could not be parsed or failed verification.

    A :class:`~kuhaku.core.exceptions.KuhakuError`, and still a ``ValueError`` as
    before. :meth:`JWTAuthProvider.authenticate` never lets it escape.
    """


_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]*$")


def _b64url_decode(segment: str) -> bytes:
    """Strict base64url: URL-safe alphabet only, no padding characters, and canonical
    (unused trailing bits zero), so exactly one encoding decodes to a given value. A
    lenient decoder accepts junk-padded or bit-flipped signatures."""

    if not _B64URL_RE.match(segment) or len(segment) % 4 == 1:
        raise JWTError("malformed token: invalid base64url segment")
    padded = segment + "=" * (-len(segment) % 4)
    try:
        data = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise JWTError("malformed token: invalid base64url segment") from exc
    if base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii") != segment:
        raise JWTError("malformed token: non-canonical base64url segment")
    return data


def _numeric_claim(payload: dict[str, Any], name: str) -> float | None:
    if name not in payload:
        return None
    value = payload[name]
    # bool is an int subclass; `true` is not a timestamp.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise JWTError(f"claim '{name}' must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise JWTError(f"claim '{name}' must be finite")
    return number


def _string_list_claim(payload: dict[str, Any], name: str) -> tuple[str, ...]:
    value = payload.get(name)
    if value is None:
        return ()
    # A bare string is refused rather than wrapped: tuple("admin") would grant the roles
    # 'a', 'd', 'm', 'i', 'n'.
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise JWTError(f"claim '{name}' must be a list of strings")
    return tuple(value)


class JWTAuthProvider:
    """Validates a JWT locally using a shared HMAC secret (HS256) -- standard library
    only (``hmac``/``hashlib``/``base64``/``json``), no external JWT dependency.

    Verification is strict: a canonical base64url signature that matches; a JSON-object
    header with the configured ``alg`` and a JSON-object payload; ``exp`` required
    (``require_exp=True``) and, like ``nbf``/``iat``, a finite number when present;
    ``aud`` checked against ``audience`` -- a token carrying an ``aud`` is refused when
    no ``audience`` is configured, because it was minted for someone else; ``iss``
    checked against ``issuer`` when configured. ``leeway`` seconds of clock skew are
    tolerated on the time claims. Identity/roles/permissions come from configurable
    claim names (default ``"sub"``/``"roles"``/``"permissions"``); the identity must be
    a non-empty string and roles/permissions lists of strings.

    Any failure, including input that is not even well formed, yields
    :meth:`AuthContext.anonymous`; :meth:`authenticate` never raises.
    """

    _REGISTERED_CLAIMS = frozenset({"exp", "nbf", "iat", "aud", "iss"})

    def __init__(
        self,
        secret: str,
        *,
        algorithm: str = "HS256",
        identity_claim: str = "sub",
        roles_claim: str = "roles",
        permissions_claim: str = "permissions",
        audience: str | Sequence[str] | None = None,
        issuer: str | None = None,
        leeway: float = 0.0,
        require_exp: bool = True,
    ) -> None:
        if algorithm != "HS256":
            raise ValueError(
                f"JWTAuthProvider: unsupported algorithm '{algorithm}' -- only 'HS256' "
                "is implemented (standard-library HMAC, no external JWT dependency)."
            )
        if not secret:
            raise ValueError("JWTAuthProvider: secret must be a non-empty string")
        if not math.isfinite(leeway) or leeway < 0:
            raise ValueError("JWTAuthProvider: leeway must be a finite, non-negative number")
        self._secret = secret.encode("utf-8")
        self._algorithm = algorithm
        self._identity_claim = identity_claim
        self._roles_claim = roles_claim
        self._permissions_claim = permissions_claim
        if isinstance(audience, str):
            audience = (audience,)
        self._audience = frozenset(audience) if audience is not None else None
        self._issuer = issuer
        self._leeway = float(leeway)
        self._require_exp = require_exp

    def _verify(self, token: str) -> dict[str, Any]:
        if not token.isascii():
            raise JWTError("malformed token: non-ASCII characters")
        parts = token.split(".")
        if len(parts) != 3:
            raise JWTError("malformed token: expected header.payload.signature")
        header_b64, payload_b64, signature_b64 = parts

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        actual_sig = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise JWTError("signature verification failed")

        try:
            header = json.loads(_b64url_decode(header_b64))
            payload = json.loads(_b64url_decode(payload_b64))
        except (ValueError, UnicodeDecodeError) as exc:
            raise JWTError("malformed token: invalid header/payload encoding") from exc
        if not isinstance(header, dict) or not isinstance(payload, dict):
            raise JWTError("malformed token: header and payload must be JSON objects")

        if header.get("alg") != self._algorithm:
            raise JWTError("token alg does not match the configured algorithm")

        now = time.time()
        exp = _numeric_claim(payload, "exp")
        nbf = _numeric_claim(payload, "nbf")
        iat = _numeric_claim(payload, "iat")
        if exp is None:
            if self._require_exp:
                raise JWTError("token has no 'exp' claim")
        elif now >= exp + self._leeway:
            raise JWTError("token has expired")
        if nbf is not None and now + self._leeway < nbf:
            raise JWTError("token is not yet valid")
        if iat is not None and now + self._leeway < iat:
            raise JWTError("token was issued in the future")

        if "aud" in payload:
            aud = payload["aud"]
            token_aud = [aud] if isinstance(aud, str) else aud
            if not isinstance(token_aud, list) or not all(isinstance(a, str) for a in token_aud):
                raise JWTError("claim 'aud' must be a string or a list of strings")
            if self._audience is None or not self._audience.intersection(token_aud):
                raise JWTError("token audience does not match")
        elif self._audience is not None:
            raise JWTError("token has no 'aud' claim")

        if self._issuer is not None and payload.get("iss") != self._issuer:
            raise JWTError("token issuer does not match")

        return payload

    def _context(self, payload: dict[str, Any]) -> AuthContext:
        identity = payload.get(self._identity_claim)
        if not isinstance(identity, str) or not identity:
            raise JWTError(f"claim '{self._identity_claim}' must be a non-empty string")
        roles = _string_list_claim(payload, self._roles_claim)
        permissions = _string_list_claim(payload, self._permissions_claim)
        reserved = {
            self._identity_claim,
            self._roles_claim,
            self._permissions_claim,
            *self._REGISTERED_CLAIMS,
        }
        metadata = {k: v for k, v in payload.items() if k not in reserved}
        return AuthContext(
            identity=identity,
            is_authenticated=True,
            roles=roles,
            permissions=permissions,
            metadata=metadata,
        )

    def authenticate(self, credential: Any) -> AuthContext:
        if not isinstance(credential, str) or not credential:
            return AuthContext.anonymous()

        token = credential.removeprefix("Bearer ")
        try:
            return self._context(self._verify(token))
        except JWTError:
            return AuthContext.anonymous()
        except Exception:
            # A hostile token must never become an exception (an HTTP 500 in serving).
            logger.debug("JWT rejected by an unexpected error", exc_info=True)
            return AuthContext.anonymous()


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
