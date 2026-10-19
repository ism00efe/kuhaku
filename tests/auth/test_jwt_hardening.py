"""Regression tests for the JWT verification hardening (security review)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from kuhaku.core.auth import AuthContext, JWTAuthProvider, JWTError
from kuhaku.core.exceptions import KuhakuError

from .test_providers import SECRET, _b64url, _make_jwt

ANON = AuthContext.anonymous()


def _raw_jwt(header: object, payload: object, secret: str = SECRET) -> str:
    header_b64 = _b64url(json.dumps(header).encode())
    payload_b64 = _b64url(json.dumps(payload).encode())
    signing_input = f"{header_b64}.{payload_b64}"
    signature = hmac.new(secret.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


def test_exp_is_required_by_default():
    token = _make_jwt({"sub": "u1"}, SECRET, exp=False)
    assert JWTAuthProvider(SECRET).authenticate(token) == ANON
    assert JWTAuthProvider(SECRET, require_exp=False).authenticate(token).identity == "u1"


@pytest.mark.parametrize("exp", [float("nan"), float("inf"), "soon", True, None, [1]])
def test_non_numeric_or_non_finite_exp_is_rejected(exp):
    # json.dumps writes NaN/Infinity as bare tokens, which json.loads accepts.
    token = _raw_jwt({"alg": "HS256"}, {"sub": "u1", "exp": exp})
    assert JWTAuthProvider(SECRET).authenticate(token) == ANON


def test_future_nbf_and_iat_are_rejected_within_leeway_rules():
    nbf = _make_jwt({"sub": "u1", "nbf": time.time() + 30}, SECRET)
    iat = _make_jwt({"sub": "u1", "iat": time.time() + 30}, SECRET)
    assert JWTAuthProvider(SECRET).authenticate(nbf) == ANON
    assert JWTAuthProvider(SECRET).authenticate(iat) == ANON
    assert JWTAuthProvider(SECRET, leeway=60).authenticate(nbf).identity == "u1"


def test_leeway_applies_to_exp():
    token = _make_jwt({"sub": "u1", "exp": time.time() - 5}, SECRET)
    assert JWTAuthProvider(SECRET).authenticate(token) == ANON
    assert JWTAuthProvider(SECRET, leeway=30).authenticate(token).identity == "u1"


def test_negative_or_non_finite_leeway_is_refused():
    with pytest.raises(ValueError):
        JWTAuthProvider(SECRET, leeway=-1)
    with pytest.raises(ValueError):
        JWTAuthProvider(SECRET, leeway=float("nan"))


def test_foreign_audience_is_rejected_unless_configured():
    token = _make_jwt({"sub": "u1", "aud": "other-service"}, SECRET)
    assert JWTAuthProvider(SECRET).authenticate(token) == ANON
    assert JWTAuthProvider(SECRET, audience="kuhaku").authenticate(token) == ANON
    assert JWTAuthProvider(SECRET, audience="other-service").authenticate(token).identity == "u1"
    listed = _make_jwt({"sub": "u1", "aud": ["a", "kuhaku"]}, SECRET)
    assert JWTAuthProvider(SECRET, audience=["kuhaku"]).authenticate(listed).identity == "u1"
    missing = _make_jwt({"sub": "u1"}, SECRET)
    assert JWTAuthProvider(SECRET, audience="kuhaku").authenticate(missing) == ANON
    mistyped = _make_jwt({"sub": "u1", "aud": 5}, SECRET)
    assert JWTAuthProvider(SECRET, audience="kuhaku").authenticate(mistyped) == ANON


def test_issuer_is_checked_when_configured():
    bad = _make_jwt({"sub": "u1", "iss": "evil"}, SECRET)
    good = _make_jwt({"sub": "u1", "iss": "idp"}, SECRET)
    assert JWTAuthProvider(SECRET, issuer="idp").authenticate(bad) == ANON
    assert JWTAuthProvider(SECRET, issuer="idp").authenticate(good).identity == "u1"


def test_junk_padded_or_non_canonical_signature_is_rejected():
    token = _make_jwt({"sub": "u1"}, SECRET)
    provider = JWTAuthProvider(SECRET)
    assert provider.authenticate(token).identity == "u1"
    head, payload, sig = token.split(".")
    for bad in (sig + "!!", sig + "==", sig[:10] + "$" + sig[10:], sig + "A", sig + " "):
        assert provider.authenticate(f"{head}.{payload}.{bad}") == ANON
    # Flip the unused low bit of the last character: a lenient decoder yields the same
    # bytes, so only a canonical-encoding check catches it.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    twin = sig[:-1] + alphabet[alphabet.index(sig[-1]) ^ 1]
    assert twin != sig
    assert provider.authenticate(f"{head}.{payload}.{twin}") == ANON


def test_string_roles_claim_is_rejected_not_split_into_letters():
    token = _make_jwt({"sub": "u1", "roles": "admin"}, SECRET)
    assert JWTAuthProvider(SECRET).authenticate(token) == ANON


@pytest.mark.parametrize(
    "payload",
    [
        {"sub": "u1", "roles": 5},
        {"sub": "u1", "permissions": {"a": 1}},
        {"sub": "u1", "roles": ["ok", 3]},
        {"sub": 42},
        {"sub": ""},
    ],
)
def test_mistyped_claims_are_rejected(payload):
    assert JWTAuthProvider(SECRET).authenticate(_make_jwt(payload, SECRET)) == ANON


@pytest.mark.parametrize("payload", [[1, 2], "str", 7, None])
def test_non_object_payload_is_rejected(payload):
    assert JWTAuthProvider(SECRET).authenticate(_raw_jwt({"alg": "HS256"}, payload)) == ANON


def test_non_object_header_is_rejected():
    assert JWTAuthProvider(SECRET).authenticate(_raw_jwt(["HS256"], {"sub": "u1"})) == ANON


@pytest.mark.parametrize(
    "token", ["é.é.é", "a.b", "a.b.c.d", "....", "\x00.\x00.\x00", "ab.cd.ef", "Bearer ", "."]
)
def test_malformed_input_never_raises(token):
    assert JWTAuthProvider(SECRET).authenticate(token) == ANON


def test_registered_claims_stay_out_of_metadata():
    token = _make_jwt({"sub": "u1", "iat": time.time() - 1, "tenant": "acme"}, SECRET)
    assert JWTAuthProvider(SECRET).authenticate(token).metadata == {"tenant": "acme"}


def test_jwt_error_is_a_kuhaku_error_and_still_a_value_error():
    assert issubclass(JWTError, KuhakuError)
    assert issubclass(JWTError, ValueError)
