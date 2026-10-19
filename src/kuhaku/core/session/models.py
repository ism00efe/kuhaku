"""The :class:`Session` working copy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..auth import AuthContext
from ..models import Message


@dataclass
class Session:
    """One conversation's state: its messages, per-tool state and metadata.

    A ``Session`` is a working copy. Change it freely, then persist it with
    :meth:`SessionStore.save`; nothing is stored until then. ``version`` is what the
    store compares on save to detect a concurrent turn (see
    :class:`~kuhaku.core.session.SessionConflict`); it and the timestamps are maintained
    by the store, not by callers.

    ``owner``/``authenticated`` record the :class:`~kuhaku.core.auth.AuthContext` that
    created the session. Both must match for any later access: an unauthenticated
    context claiming the same identity string is a different owner.

    ``state`` is namespaced by tool -- ``session.tool_state("extraction")`` -- so two
    tools sharing one conversation never overwrite each other's keys. Everything in it
    must be JSON-serializable.
    """

    id: str
    owner: str
    authenticated: bool
    created_at: float
    updated_at: float
    version: int = 0
    expires_at: float | None = None
    ttl_seconds: float | None = None
    messages: list[Message] = field(default_factory=list)
    state: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, *messages: Message) -> None:
        for message in messages:
            if not isinstance(message, Message):
                raise TypeError(f"Expected a Message, got {type(message).__name__}.")
            self.messages.append(message)

    def tool_state(self, tool: str) -> dict[str, Any]:
        """This tool's state dict, created empty on first use. Mutate it in place."""

        if not tool:
            raise ValueError("tool must be a non-empty namespace name.")
        return self.state.setdefault(tool, {})

    def clear_tool_state(self, tool: str) -> None:
        self.state.pop(tool, None)

    def is_owned_by(self, auth: AuthContext) -> bool:
        return self.owner == auth.identity and self.authenticated == auth.is_authenticated
