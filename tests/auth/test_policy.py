"""Tests for AuthorizationPolicy, DefaultAllowPolicy, and ConfigAuthorizationPolicy."""

from __future__ import annotations

import json

import pytest

from kuhaku.core.auth import AuthContext, AuthorizationPolicy
from kuhaku.core.auth.policy import (
    ConfigAuthorizationPolicy,
    DefaultAllowPolicy,
    PolicyConfigError,
)


def _ctx(*, roles=(), permissions=()) -> AuthContext:
    return AuthContext(identity="u1", is_authenticated=True, roles=roles, permissions=permissions)


# --- DefaultAllowPolicy ------------------------------------------------------------
def test_default_allow_policy_allows_everything():
    policy = DefaultAllowPolicy()
    assert policy.check(AuthContext.anonymous(), "document", "read") is True
    assert policy.check(_ctx(), "anything", "delete") is True


def test_default_allow_policy_satisfies_the_protocol():
    assert isinstance(DefaultAllowPolicy(), AuthorizationPolicy)


def test_config_authorization_policy_satisfies_the_protocol():
    assert isinstance(ConfigAuthorizationPolicy(), AuthorizationPolicy)


# --- ConfigAuthorizationPolicy: programmatic role/rule management ------------------
def test_no_rule_configured_denies_by_default():
    policy = ConfigAuthorizationPolicy()
    assert policy.check(_ctx(roles=("editor",)), "document", "read") is False


def test_role_with_matching_rule_is_allowed():
    policy = ConfigAuthorizationPolicy()
    policy.add_role("viewer", ["document:read"])
    policy.set_rule("document", "read", allowed_roles=["viewer"])
    assert policy.check(_ctx(roles=("viewer",)), "document", "read") is True


def test_role_without_matching_rule_is_denied():
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("document", "read", allowed_roles=["viewer"])
    assert policy.check(_ctx(roles=("editor",)), "document", "read") is False


def test_permission_listed_directly_on_the_rule_is_allowed():
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("document", "read", allowed_roles=[], allowed_permissions=["document:read"])
    assert policy.check(_ctx(permissions=("document:read",)), "document", "read") is True


def test_permission_granted_via_role_is_allowed():
    policy = ConfigAuthorizationPolicy()
    policy.add_role("editor", ["document:read", "document:write"])
    policy.set_rule("document", "read", allowed_roles=[], allowed_permissions=["document:read"])
    assert policy.check(_ctx(roles=("editor",)), "document", "read") is True


def test_grant_permission_defines_role_implicitly():
    policy = ConfigAuthorizationPolicy()
    policy.grant_permission("editor", "document:read")
    policy.set_rule("document", "read", allowed_roles=[], allowed_permissions=["document:read"])
    assert policy.check(_ctx(roles=("editor",)), "document", "read") is True


def test_revoke_permission_removes_access():
    policy = ConfigAuthorizationPolicy()
    policy.add_role("editor", ["document:read"])
    policy.set_rule("document", "read", allowed_roles=[], allowed_permissions=["document:read"])
    policy.revoke_permission("editor", "document:read")
    assert policy.check(_ctx(roles=("editor",)), "document", "read") is False


def test_revoke_permission_on_unknown_role_raises():
    policy = ConfigAuthorizationPolicy()
    with pytest.raises(PolicyConfigError):
        policy.revoke_permission("ghost", "document:read")


def test_remove_role_revokes_the_permissions_it_granted():
    """remove_role clears what the role grants via the permission path -- it does not
    retroactively edit a rule's own `allowed_roles` list (that's set_rule's job; the two
    are independent, see ConfigAuthorizationPolicy's docstring)."""

    policy = ConfigAuthorizationPolicy()
    policy.add_role("editor", ["document:read"])
    policy.set_rule("document", "read", allowed_roles=[], allowed_permissions=["document:read"])
    assert policy.check(_ctx(roles=("editor",)), "document", "read") is True

    policy.remove_role("editor")
    assert policy.check(_ctx(roles=("editor",)), "document", "read") is False


def test_remove_role_is_a_noop_for_an_unknown_role():
    policy = ConfigAuthorizationPolicy()
    policy.remove_role("never-existed")  # must not raise


def test_add_role_replaces_prior_definition():
    policy = ConfigAuthorizationPolicy()
    policy.add_role("editor", ["document:read"])
    policy.add_role("editor", ["document:write"])  # replaces, does not merge
    policy.set_rule("document", "read", allowed_roles=[], allowed_permissions=["document:read"])
    assert policy.check(_ctx(roles=("editor",)), "document", "read") is False


def test_set_rule_replaces_prior_rule_for_the_same_pair():
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("document", "read", allowed_roles=["viewer"])
    policy.set_rule("document", "read", allowed_roles=["editor"])
    assert policy.check(_ctx(roles=("viewer",)), "document", "read") is False
    assert policy.check(_ctx(roles=("editor",)), "document", "read") is True


def test_different_resource_action_pairs_are_independent():
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("document", "read", allowed_roles=["viewer"])
    assert policy.check(_ctx(roles=("viewer",)), "document", "write") is False
    assert policy.check(_ctx(roles=("viewer",)), "log", "read") is False


# --- wildcards -----------------------------------------------------------------
def test_wildcard_role_on_rule_allows_any_role():
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("document", "read", allowed_roles=["*"])
    assert policy.check(_ctx(roles=("literally-anything",)), "document", "read") is True


def test_wildcard_permission_on_rule_allows_any_permission():
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("document", "read", allowed_roles=[], allowed_permissions=["*"])
    assert policy.check(_ctx(permissions=("literally-anything",)), "document", "read") is True


def test_wildcard_role_on_rule_matches_unconditionally_even_anonymous():
    """`"*"` on a rule means "anyone, no restriction" for that resource/action pair --
    equivalent to DefaultAllowPolicy scoped to just that pair, including a context with
    no roles/permissions at all (see ConfigAuthorizationPolicy's docstring)."""

    policy = ConfigAuthorizationPolicy()
    policy.set_rule("document", "read", allowed_roles=["*"])
    assert policy.check(AuthContext.anonymous(), "document", "read") is True


def test_wildcard_permission_on_a_role_grants_every_rule_it_matches():
    policy = ConfigAuthorizationPolicy()
    policy.add_role("superuser", ["*"])
    policy.set_rule("document", "read", allowed_roles=[], allowed_permissions=["document:read"])
    assert policy.check(_ctx(roles=("superuser",)), "document", "read") is True


# --- error messages --------------------------------------------------------------
def test_add_role_rejects_empty_role_name():
    with pytest.raises(PolicyConfigError, match="role"):
        ConfigAuthorizationPolicy().add_role("", ["x"])


def test_set_rule_rejects_empty_resource_or_action():
    policy = ConfigAuthorizationPolicy()
    with pytest.raises(PolicyConfigError, match="resource"):
        policy.set_rule("", "read", allowed_roles=["viewer"])
    with pytest.raises(PolicyConfigError, match="resource"):
        policy.set_rule("document", "", allowed_roles=["viewer"])


# --- file/dict-based configuration -------------------------------------------------
_CONFIG = {
    "roles": {"editor": ["document:write"], "viewer": ["document:read"]},
    "rules": [
        {"resource": "document", "action": "read", "allowed_roles": ["viewer", "editor"]},
        {"resource": "document", "action": "write", "allowed_roles": ["editor"]},
    ],
}


def test_load_dict_wires_up_roles_and_rules():
    policy = ConfigAuthorizationPolicy()
    policy.load_dict(_CONFIG)

    assert policy.check(_ctx(roles=("viewer",)), "document", "read") is True
    assert policy.check(_ctx(roles=("viewer",)), "document", "write") is False
    assert policy.check(_ctx(roles=("editor",)), "document", "write") is True


def test_load_dict_replaces_prior_configuration_entirely():
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("document", "read", allowed_roles=["someone-else"])
    policy.load_dict(_CONFIG)
    assert policy.check(_ctx(roles=("someone-else",)), "document", "read") is False
    assert policy.check(_ctx(roles=("viewer",)), "document", "read") is True


def test_load_file_json(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_CONFIG), encoding="utf-8")

    policy = ConfigAuthorizationPolicy()
    policy.load_file(path)
    assert policy.check(_ctx(roles=("viewer",)), "document", "read") is True


def test_load_file_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(PolicyConfigError, match="no such file"):
        ConfigAuthorizationPolicy().load_file(tmp_path / "nope.json")


def test_load_file_invalid_json_raises_clear_error(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(PolicyConfigError, match="invalid JSON"):
        ConfigAuthorizationPolicy().load_file(path)


def test_load_dict_rejects_non_mapping_top_level():
    with pytest.raises(PolicyConfigError, match="mapping"):
        ConfigAuthorizationPolicy().load_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_load_dict_rejects_role_permissions_that_are_not_a_list():
    with pytest.raises(PolicyConfigError, match="roles.editor"):
        ConfigAuthorizationPolicy().load_dict({"roles": {"editor": "document:read"}})


def test_load_dict_rejects_rule_missing_resource_or_action():
    with pytest.raises(PolicyConfigError, match="rules\\[0\\]"):
        ConfigAuthorizationPolicy().load_dict({"rules": [{"action": "read"}]})


def test_load_file_yaml_without_pyyaml_installed_raises_clear_error(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no module named yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    path = tmp_path / "policy.yaml"
    path.write_text("roles: {}\nrules: []\n", encoding="utf-8")
    with pytest.raises(PolicyConfigError, match="PyYAML"):
        ConfigAuthorizationPolicy().load_file(path)
