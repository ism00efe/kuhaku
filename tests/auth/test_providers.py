"""Tests for AuthProvider implementations and AuthProviderRegistry."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

from kuhaku.core.auth import AuthContext, AuthProvider
from kuhaku.core.auth.providers import (
    APIKeyAuthProvider,
    AuthProviderRegistry,
    JWTAuthProvider,
    NoOpAuthProvider,
)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_jwt(payload: dict, secret: str, *, alg: str = "HS256") -> str:
    header = {"alg": alg, "typ": "JWT"}
    header_b64 = _b64url(json.dumps(header).encode())
    payload_b64 = _b64url(json.dumps(payload).encode())
    signing_input = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64url(signature)}"


# --- NoOpAuthProvider ---------------------------------------------------------------
def test_noop_provider_always_returns_anonymous():
    provider = NoOpAuthProvider()
    assert provider.authenticate("literally anything") == AuthContext.anonymous()
    assert provider.authenticate(None) == AuthContext.anonymous()


def test_noop_provider_satisfies_the_protocol():
    assert isinstance(NoOpAuthProvider(), AuthProvider)


# --- APIKeyAuthProvider --------------------------------------------------------------
def test_api_key_provider_authenticates_a_registered_key():
    provider = APIKeyAuthProvider()
    provider.add_key("secret-key-1", identity="svc-a", roles=["service"])

    ctx = provider.authenticate("secret-key-1")
    assert ctx.identity == "svc-a"
    assert ctx.is_authenticated is True
    assert ctx.roles == ("service",)


def test_api_key_provider_identity_defaults_to_the_key_itself():
    provider = APIKeyAuthProvider()
    provider.add_key("secret-key-1")
    assert provider.authenticate("secret-key-1").identity == "secret-key-1"


def test_api_key_provider_rejects_unknown_key():
    provider = APIKeyAuthProvider()
    provider.add_key("secret-key-1")
    assert provider.authenticate("wrong-key") == AuthContext.anonymous()


def test_api_key_provider_rejects_non_string_credential():
    provider = APIKeyAuthProvider()
    provider.add_key("secret-key-1")
    assert provider.authenticate(12345) == AuthContext.anonymous()


def test_api_key_provider_constructor_accepts_a_keys_table():
    provider = APIKeyAuthProvider({"k1": {"identity": "u1", "roles": ["viewer"]}})
    ctx = provider.authenticate("k1")
    assert ctx.identity == "u1" and ctx.roles == ("viewer",)


def test_api_key_provider_remove_key_revokes_access():
    provider = APIKeyAuthProvider()
    provider.add_key("secret-key-1")
    provider.remove_key("secret-key-1")
    assert provider.authenticate("secret-key-1") == AuthContext.anonymous()


def test_api_key_provider_add_key_rejects_empty_key():
    with pytest.raises(ValueError):
        APIKeyAuthProvider().add_key("")


# --- JWTAuthProvider ------------------------------------------------------------------
SECRET = "test-shared-secret"


def test_jwt_provider_authenticates_a_valid_token():
    provider = JWTAuthProvider(SECRET)
    token = _make_jwt({"sub": "u1", "roles": ["editor"], "permissions": ["document:write"]}, SECRET)

    ctx = provider.authenticate(token)
    assert ctx.identity == "u1"
    assert ctx.is_authenticated is True
    assert ctx.roles == ("editor",)
    assert ctx.permissions == ("document:write",)


def test_jwt_provider_accepts_bearer_prefix():
    provider = JWTAuthProvider(SECRET)
    token = _make_jwt({"sub": "u1"}, SECRET)
    ctx = provider.authenticate(f"Bearer {token}")
    assert ctx.identity == "u1"


def test_jwt_provider_rejects_wrong_secret():
    provider = JWTAuthProvider(SECRET)
    token = _make_jwt({"sub": "u1"}, "a-different-secret")
    assert provider.authenticate(token) == AuthContext.anonymous()


def test_jwt_provider_rejects_expired_token():
    provider = JWTAuthProvider(SECRET)
    token = _make_jwt({"sub": "u1", "exp": time.time() - 60}, SECRET)
    assert provider.authenticate(token) == AuthContext.anonymous()


def test_jwt_provider_accepts_unexpired_token():
    provider = JWTAuthProvider(SECRET)
    token = _make_jwt({"sub": "u1", "exp": time.time() + 60}, SECRET)
    assert provider.authenticate(token).identity == "u1"


def test_jwt_provider_rejects_malformed_token():
    provider = JWTAuthProvider(SECRET)
    assert provider.authenticate("not-a-jwt") == AuthContext.anonymous()


def test_jwt_provider_rejects_tampered_payload():
    provider = JWTAuthProvider(SECRET)
    token = _make_jwt({"sub": "u1"}, SECRET)
    header_b64, payload_b64, sig_b64 = token.split(".")
    tampered_payload = _b64url(json.dumps({"sub": "attacker"}).encode())
    tampered = f"{header_b64}.{tampered_payload}.{sig_b64}"
    assert provider.authenticate(tampered) == AuthContext.anonymous()


def test_jwt_provider_rejects_non_string_credential():
    provider = JWTAuthProvider(SECRET)
    assert provider.authenticate(None) == AuthContext.anonymous()


def test_jwt_provider_rejects_token_with_no_identity_claim():
    provider = JWTAuthProvider(SECRET)
    token = _make_jwt({"roles": ["editor"]}, SECRET)  # no "sub"
    assert provider.authenticate(token) == AuthContext.anonymous()


def test_jwt_provider_supports_custom_claim_names():
    provider = JWTAuthProvider(SECRET, identity_claim="user_id", roles_claim="grp")
    token = _make_jwt({"user_id": "u1", "grp": ["admin"]}, SECRET)
    ctx = provider.authenticate(token)
    assert ctx.identity == "u1"
    assert ctx.roles == ("admin",)


def test_jwt_provider_extra_claims_land_in_metadata():
    provider = JWTAuthProvider(SECRET)
    token = _make_jwt({"sub": "u1", "tenant": "acme"}, SECRET)
    assert provider.authenticate(token).metadata == {"tenant": "acme"}


def test_jwt_provider_rejects_empty_secret():
    with pytest.raises(ValueError):
        JWTAuthProvider("")


def test_jwt_provider_rejects_unsupported_algorithm():
    with pytest.raises(ValueError, match="HS256"):
        JWTAuthProvider(SECRET, algorithm="RS256")


def test_jwt_provider_rejects_token_signed_with_a_different_alg_header():
    provider = JWTAuthProvider(SECRET)
    token = _make_jwt({"sub": "u1"}, SECRET, alg="none")
    assert provider.authenticate(token) == AuthContext.anonymous()


# --- AuthProviderRegistry -------------------------------------------------------------
def test_registry_defaults_to_noop():
    registry = AuthProviderRegistry()
    assert registry.authenticate("whatever") == AuthContext.anonymous()


def test_registry_register_and_resolve_by_name():
    registry = AuthProviderRegistry()
    provider = APIKeyAuthProvider({"k1": {"identity": "u1"}})
    registry.register("apikey", provider)
    assert registry.get("apikey") is provider
    assert registry.authenticate("k1", provider="apikey").identity == "u1"


def test_registry_register_with_make_default_switches_default():
    registry = AuthProviderRegistry()
    provider = APIKeyAuthProvider({"k1": {"identity": "u1"}})
    registry.register("apikey", provider, make_default=True)
    assert registry.authenticate("k1").identity == "u1"


def test_registry_get_unknown_name_raises():
    with pytest.raises(KeyError):
        AuthProviderRegistry().get("nope")


def test_registry_set_default_switches_which_provider_authenticate_uses():
    registry = AuthProviderRegistry()
    provider = APIKeyAuthProvider({"k1": {"identity": "u1"}})
    registry.register("apikey", provider)
    registry.set_default("apikey")
    assert registry.authenticate("k1").identity == "u1"


def test_registry_set_default_unknown_name_raises():
    with pytest.raises(KeyError):
        AuthProviderRegistry().set_default("nope")


def test_registry_unregister_removes_provider():
    registry = AuthProviderRegistry()
    registry.register("apikey", APIKeyAuthProvider())
    registry.unregister("apikey")
    with pytest.raises(KeyError):
        registry.get("apikey")


def test_registry_unregister_current_default_raises():
    registry = AuthProviderRegistry()
    registry.register("apikey", APIKeyAuthProvider(), make_default=True)
    with pytest.raises(ValueError, match="default"):
        registry.unregister("apikey")


def test_registry_register_rejects_empty_name():
    with pytest.raises(ValueError):
        AuthProviderRegistry().register("", NoOpAuthProvider())
