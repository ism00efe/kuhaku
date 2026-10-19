"""Turning a request's headers into an :class:`~kuhaku.core.auth.AuthContext`.

kuhaku still authenticates no one by itself: :class:`RequestAuthenticator` only reads
one header and hands the credential to a caller-configured
:class:`~kuhaku.core.auth.AuthProvider` (``APIKeyAuthProvider``, ``JWTAuthProvider``, or
the application's own). Identity and roles come from that provider and nothing else --
request bodies are never consulted, and the wire schemas reject unknown fields, so a
body carrying ``roles`` is a 422, not a silent no-op.
"""

from __future__ import annotations

from collections.abc import Mapping

from kuhaku.core.auth import AuthContext, AuthProvider

from .errors import AuthenticationRequired


class RequestAuthenticator:
    """Authenticates one request from its headers.

    ``header``/``scheme`` say where the credential is: the default reads
    ``Authorization: Bearer <credential>``, which suits both JWTs and API keys; pass
    ``header="X-API-Key", scheme=None`` for a bare key header.

    ``required`` decides what a request *without* a credential gets: ``True`` (the
    default) rejects it (401); ``False`` lets it through as
    :meth:`AuthContext.anonymous`, which sees only untagged documents. A credential that
    is present but does not validate is rejected either way -- quietly downgrading a
    wrong or expired token to anonymous would hide the mistake from the client.
    """

    def __init__(
        self,
        provider: AuthProvider,
        *,
        required: bool = True,
        header: str = "Authorization",
        scheme: str | None = "Bearer",
    ) -> None:
        if not header:
            raise ValueError("RequestAuthenticator: header must be a non-empty string")
        self.provider = provider
        self.required = required
        self.header = header
        self.scheme = scheme

    @property
    def challenge(self) -> str:
        """The ``WWW-Authenticate`` value sent with a 401."""

        return self.scheme or "ApiKey"

    def credential(self, headers: Mapping[str, str]) -> str | None:
        """The credential in ``headers``, or ``None`` when there is none.

        Header names are matched case-insensitively whatever mapping is passed in. A
        header present with the wrong scheme counts as a malformed credential, not as
        none, so it is rejected rather than treated as anonymous.
        """

        wanted = self.header.lower()
        raw = next((v for k, v in headers.items() if k.lower() == wanted), None)
        if raw is None or not raw.strip():
            return None
        raw = raw.strip()
        if self.scheme is None:
            return raw
        prefix, _, value = raw.partition(" ")
        if prefix.lower() != self.scheme.lower() or not value.strip():
            raise AuthenticationRequired("Malformed credential.")
        return value.strip()

    def authenticate(self, headers: Mapping[str, str]) -> AuthContext:
        """The :class:`AuthContext` for a request with ``headers``.

        Raises :class:`~kuhaku.serving.errors.AuthenticationRequired` when the request
        must be rejected.
        """

        credential = self.credential(headers)
        if credential is None:
            if self.required:
                raise AuthenticationRequired()
            return AuthContext.anonymous()
        try:
            context = self.provider.authenticate(credential)
        except Exception as exc:
            # The AuthProvider contract is "never raise"; a provider that does anyway
            # is treated as a rejected credential, not turned into a 500 that would
            # distinguish hostile input from a wrong key.
            raise AuthenticationRequired("The credential is invalid or expired.") from exc
        if not isinstance(context, AuthContext) or not context.is_authenticated:
            raise AuthenticationRequired("The credential is invalid or expired.")
        return context
