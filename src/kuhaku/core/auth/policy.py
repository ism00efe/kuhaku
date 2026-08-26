"""Authorization: a pluggable ``AuthorizationPolicy`` decides whether an ``AuthContext``
may perform an ``action`` on a ``resource``. Two implementations ship here:
``DefaultAllowPolicy`` (allows everything -- a caller can wire this in explicitly for a
self-documenting "no restrictions" policy) and ``ConfigAuthorizationPolicy`` (role/
permission rules, defined by the application via a file or programmatically at runtime --
never a hierarchy kuhaku itself imposes).

Neither is wired into ``RAGEngine`` at all -- it has no ``authorization_policy``
constructor parameter or call site of any kind. The only access control kuhaku enforces
itself is document-level tag filtering (``access_tags``/``AuthContext.roles``, see
``kuhaku.tools.rag.retriever.is_entitled``); resource/action-level checks via
``AuthorizationPolicy.check()`` are a caller's responsibility to call directly (e.g.
before invoking ``RAG.ask()``), not something this package calls for you today.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .context import AuthContext

_WILDCARD = "*"


class PolicyConfigError(ValueError):
    """A ``ConfigAuthorizationPolicy`` configuration file or call was invalid."""


@runtime_checkable
class AuthorizationPolicy(Protocol):
    """Decides whether ``context`` may perform ``action`` on ``resource``.

    ``resource``/``action`` are plain strings the caller defines the vocabulary for (e.g.
    ``"document"``/``"read"``) -- kuhaku does not constrain their values.
    """

    def check(self, context: AuthContext, resource: str, action: str) -> bool: ...


class DefaultAllowPolicy:
    """Allows every check unconditionally.

    Behaviorally identical to leaving ``authorization_policy`` unset (``None``) -- this
    class exists for callers who want an explicit, self-documenting "no restrictions"
    policy object instead of writing their own no-op ``check()``.
    """

    def check(self, context: AuthContext, resource: str, action: str) -> bool:
        return True


class ConfigAuthorizationPolicy:
    """Role/permission-based ``AuthorizationPolicy``, defined entirely by the caller.

    Two kinds of rule feed a decision:

    - role -> permissions (:meth:`add_role`/:meth:`grant_permission`): what each role
      can do.
    - resource/action -> allowed roles/permissions (:meth:`set_rule`): what's needed to
      perform one specific action on one specific resource.

    :meth:`check` allows access when the context carries a role or permission listed
    directly on the ``(resource, action)`` rule, or a permission granted to one of the
    context's roles is listed on that rule. ``"*"`` in a rule's allowed-roles or
    allowed-permissions list matches unconditionally (anyone); ``"*"`` among a role's own
    permissions grants that role every permission any rule might ask for.

    A ``(resource, action)`` pair with no rule configured at all has nothing to check
    against and is denied -- this policy is opt-in, default-deny-by-rule once configured.
    Callers wanting default-allow should not configure a policy at all (see
    ``DefaultAllowPolicy``/the module docstring).
    """

    def __init__(self) -> None:
        self._roles: dict[str, set[str]] = {}
        self._rules: dict[tuple[str, str], tuple[set[str], set[str]]] = {}

    # --- programmatic configuration ---------------------------------------------

    def add_role(self, role: str, permissions: list[str] | tuple[str, ...] = ()) -> None:
        """Define ``role``, granting it ``permissions`` (replaces any existing definition
        for the same role)."""

        if not role:
            raise PolicyConfigError("add_role: role must be a non-empty string")
        self._roles[role] = set(permissions)

    def remove_role(self, role: str) -> None:
        """Remove ``role`` entirely. A no-op if ``role`` was never defined."""

        self._roles.pop(role, None)

    def grant_permission(self, role: str, permission: str) -> None:
        """Add ``permission`` to ``role``, defining ``role`` first if it doesn't exist."""

        if not role:
            raise PolicyConfigError("grant_permission: role must be a non-empty string")
        self._roles.setdefault(role, set()).add(permission)

    def revoke_permission(self, role: str, permission: str) -> None:
        """Remove ``permission`` from ``role``.

        Raises if ``role`` is undefined -- revoking a permission from a role that was
        never granted anything is almost always a caller mistake, not a no-op.
        """

        if role not in self._roles:
            raise PolicyConfigError(f"revoke_permission: unknown role '{role}'")
        self._roles[role].discard(permission)

    def set_rule(
        self,
        resource: str,
        action: str,
        allowed_roles: list[str] | tuple[str, ...],
        allowed_permissions: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Define who may perform ``action`` on ``resource``.

        ``allowed_roles``/``allowed_permissions`` each support the wildcard ``"*"`` --
        present in either list, every check against this rule passes. Replaces any
        existing rule for the same ``(resource, action)`` pair.
        """

        if not resource or not action:
            raise PolicyConfigError("set_rule: resource and action must be non-empty strings")
        self._rules[(resource, action)] = (
            set(allowed_roles),
            set(allowed_permissions or ()),
        )

    # --- file-based configuration -------------------------------------------------

    def load_file(self, path: str | Path) -> None:
        """Load role/rule definitions from a JSON or YAML file, replacing this policy's
        current configuration entirely.

        Expected shape::

            {
              "roles": {
                "editor": ["document:write"],
                "viewer": ["document:read"]
              },
              "rules": [
                {"resource": "document", "action": "read",
                 "allowed_roles": ["viewer", "editor"], "allowed_permissions": []}
              ]
            }

        The format is picked by file extension: ``.yaml``/``.yml`` parses as YAML
        (requires PyYAML to already be installed -- this module never adds it as a
        dependency); anything else parses as JSON via the standard library.
        """

        file_path = Path(path)
        if not file_path.is_file():
            raise PolicyConfigError(f"load_file: no such file '{file_path}'")

        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise PolicyConfigError(
                    f"load_file: '{file_path}' is a YAML file but PyYAML is not "
                    "installed; install it or provide a JSON file instead."
                ) from exc
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise PolicyConfigError(f"load_file: invalid YAML in '{file_path}': {exc}") from exc
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise PolicyConfigError(f"load_file: invalid JSON in '{file_path}': {exc}") from exc

        self._load_mapping(data, source=str(file_path))

    def load_dict(self, data: dict[str, Any]) -> None:
        """Load role/rule definitions from an already-parsed mapping (see
        :meth:`load_file` for the expected shape), replacing this policy's current
        configuration entirely."""

        self._load_mapping(data, source="<dict>")

    def _load_mapping(self, data: Any, *, source: str) -> None:
        if not isinstance(data, dict):
            raise PolicyConfigError(
                f"{source}: expected a top-level object/mapping, got {type(data).__name__}"
            )

        roles: dict[str, set[str]] = {}
        for role, permissions in (data.get("roles") or {}).items():
            if not isinstance(permissions, list):
                raise PolicyConfigError(
                    f"{source}: roles.{role} must be a list of permission strings"
                )
            roles[role] = set(permissions)

        rules: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
        for i, rule in enumerate(data.get("rules") or []):
            if not isinstance(rule, dict):
                raise PolicyConfigError(f"{source}: rules[{i}] must be an object")
            resource = rule.get("resource")
            action = rule.get("action")
            if not resource or not action:
                raise PolicyConfigError(
                    f"{source}: rules[{i}] must define non-empty 'resource' and 'action'"
                )
            rules[(resource, action)] = (
                set(rule.get("allowed_roles") or ()),
                set(rule.get("allowed_permissions") or ()),
            )

        self._roles = roles
        self._rules = rules

    # --- checks -----------------------------------------------------------------

    def check(self, context: AuthContext, resource: str, action: str) -> bool:
        rule = self._rules.get((resource, action))
        if rule is None:
            return False

        allowed_roles, allowed_permissions = rule
        if _WILDCARD in allowed_roles or _WILDCARD in allowed_permissions:
            return True

        if any(role in allowed_roles for role in context.roles):
            return True
        if any(permission in allowed_permissions for permission in context.permissions):
            return True

        for role in context.roles:
            role_permissions = self._roles.get(role, set())
            if _WILDCARD in role_permissions or role_permissions & allowed_permissions:
                return True

        return False
