"""Tests for AuthContext."""

from __future__ import annotations

from kuhaku.core.auth import AuthContext


def test_anonymous_has_no_identity_roles_or_permissions():
    ctx = AuthContext.anonymous()
    assert ctx.identity == ""
    assert ctx.is_authenticated is False
    assert ctx.roles == ()
    assert ctx.permissions == ()
    assert ctx.metadata == {}


def test_defaults_are_unauthenticated_and_empty():
    ctx = AuthContext(identity="u1")
    assert ctx.is_authenticated is False
    assert ctx.roles == ()
    assert ctx.permissions == ()
    assert ctx.metadata == {}


def test_fully_populated_context_round_trips_its_fields():
    ctx = AuthContext(
        identity="u1",
        is_authenticated=True,
        roles=("editor", "viewer"),
        permissions=("document:read",),
        metadata={"tenant": "acme"},
    )
    assert ctx.identity == "u1"
    assert ctx.is_authenticated is True
    assert ctx.roles == ("editor", "viewer")
    assert ctx.permissions == ("document:read",)
    assert ctx.metadata == {"tenant": "acme"}


def test_is_frozen():
    ctx = AuthContext(identity="u1")
    try:
        ctx.identity = "u2"  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("AuthContext must be immutable")


def test_two_contexts_with_equal_fields_are_equal():
    a = AuthContext(identity="u1", is_authenticated=True, roles=("viewer",))
    b = AuthContext(identity="u1", is_authenticated=True, roles=("viewer",))
    assert a == b
