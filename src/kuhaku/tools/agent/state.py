"""The resumable state of a run paused for approval.

A paused run leaves the process as data (a database row, a queue message, a hidden form
field) and comes back through :meth:`Agent.resume <kuhaku.tools.agent.Agent.resume>`.
Everything in it decides what will execute next -- the pending calls and their
arguments, whose run it is -- so it is sealed with an HMAC over its canonical JSON, and
``resume`` refuses a state whose seal does not verify. The seal makes the state
tamper-evident, not secret: it carries the conversation in clear text, so store it the
way you store the conversation itself.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

from kuhaku.core.auth import AuthContext
from kuhaku.core.exceptions import KuhakuError
from kuhaku.core.models import (
    ContentPart,
    Message,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
)

# 2: issued_at (expiry) and per-call approval digests. A version-1 state is refused.
STATE_VERSION = 2


class AgentStateError(KuhakuError):
    """A state cannot be resumed: its seal does not verify (tampered with, or sealed by
    an agent with another ``state_secret``), it belongs to another identity, it has
    expired, it was already resumed, or it is not a state at all."""


def call_digest(call: ToolCall) -> str:
    """The approval digest of ``call``: SHA-256 over its tool name and canonical JSON
    arguments. An approval is bound to this, not just to the call's id, so it can only
    ever run exactly the call a human was shown."""

    canonical = json.dumps(
        {"name": call.name, "arguments": call.arguments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentState:
    """A sealed, JSON-serializable snapshot of a paused run.

    ``to_json()`` / ``AgentState.from_json(text)`` round-trip it; the seal is checked by
    ``Agent.resume``, not here, because only the agent holds the key.
    """

    payload: dict[str, Any]
    signature: str

    @property
    def trace_id(self) -> str:
        return str(self.payload.get("trace_id", ""))

    @property
    def pending(self) -> tuple[ToolCall, ...]:
        """The calls waiting for a decision."""

        ids = set(self.pending_digests)
        last = self.messages[-1] if self.messages else None
        return tuple(c for c in (last.tool_calls if last else ()) if c.id in ids)

    @property
    def pending_digests(self) -> dict[str, str]:
        """Pending call id -> :func:`call_digest`. Show the human the call, and pass its
        digest as the approval value to approve exactly that call."""

        return {
            str(entry.get("id")): str(entry.get("digest"))
            for entry in self.payload.get("pending", ())
            if isinstance(entry, dict)
        }

    @property
    def issued_at(self) -> float | None:
        value = self.payload.get("issued_at")
        return float(value) if isinstance(value, int | float) else None

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(message_from_dict(m) for m in self.payload.get("messages", ()))

    def to_dict(self) -> dict[str, Any]:
        return {"payload": self.payload, "signature": self.signature}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Any) -> AgentState:
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("payload"), dict)
            or not isinstance(data.get("signature"), str)
        ):
            raise AgentStateError(
                "Not an agent state: expected {'payload', 'signature'}."
            )
        return cls(payload=data["payload"], signature=data["signature"])

    @classmethod
    def from_json(cls, text: str) -> AgentState:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentStateError(
                f"Not an agent state: invalid JSON ({exc.msg})."
            ) from None
        return cls.from_dict(data)


def seal(payload: dict[str, Any], key: bytes) -> AgentState:
    # Round-tripped through JSON first, so the payload kept is exactly what the
    # signature covers (a tuple becomes a list, a non-JSON value its str()).
    normalized = json.loads(_canonical(payload))
    return AgentState(payload=normalized, signature=_sign(normalized, key))


def verify(state: AgentState, key: bytes) -> None:
    if not hmac.compare_digest(state.signature, _sign(state.payload, key)):
        raise AgentStateError(
            "Agent state failed verification: it was modified, or sealed by an agent "
            "with a different state_secret."
        )
    if state.payload.get("v") != STATE_VERSION:
        raise AgentStateError(
            f"Unsupported agent state version {state.payload.get('v')!r}."
        )


def identity_of(auth_context: AuthContext | None) -> dict[str, Any] | None:
    """What a state binds to: who the caller is, not what they may do -- roles are
    re-read from the ``AuthContext`` given to ``resume``, so a role revoked while the
    run waited takes effect."""

    if auth_context is None:
        return None
    return {
        "identity": auth_context.identity,
        "is_authenticated": auth_context.is_authenticated,
    }


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def _sign(payload: dict[str, Any], key: bytes) -> str:
    return hmac.new(
        key, _canonical(payload).encode("utf-8"), hashlib.sha256
    ).hexdigest()


# --- message (de)serialization ----------------------------------------------------------


def message_to_dict(message: Message) -> dict[str, Any]:
    content: Any
    if isinstance(message.content, str):
        content = message.content
    else:
        content = [_part_to_dict(part) for part in message.parts]
    return {
        "role": message.role,
        "content": content,
        "metadata": dict(message.metadata),
    }


def message_from_dict(data: dict[str, Any]) -> Message:
    content = data["content"]
    if not isinstance(content, str):
        content = tuple(_part_from_dict(part) for part in content)
    return Message(data["role"], content, metadata=dict(data.get("metadata") or {}))


def call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {
        "name": call.name,
        "arguments": call.arguments,
        "id": call.id,
        "raw_arguments": call.raw_arguments,
        "arguments_error": call.arguments_error,
    }


def call_from_dict(data: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=data["name"],
        arguments=dict(data.get("arguments") or {}),
        id=data.get("id", ""),
        raw_arguments=data.get("raw_arguments"),
        arguments_error=data.get("arguments_error"),
    )


def _part_to_dict(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ToolCallPart):
        return {"type": "tool_call", "call": call_to_dict(part.call)}
    if isinstance(part, ToolResultPart):
        return {
            "type": "tool_result",
            "tool_call_id": part.tool_call_id,
            "content": part.content,
            "name": part.name,
            "is_error": part.is_error,
        }
    raise AgentStateError(f"Cannot serialize a message part of type {part.type!r}.")


def _part_from_dict(data: dict[str, Any]) -> ContentPart:
    kind = data.get("type")
    if kind == "text":
        return TextPart(data["text"])
    if kind == "tool_call":
        return ToolCallPart(call_from_dict(data["call"]))
    if kind == "tool_result":
        return ToolResultPart(
            data["tool_call_id"],
            data["content"],
            name=data.get("name", ""),
            is_error=bool(data.get("is_error", False)),
        )
    raise AgentStateError(f"Unknown message part type {kind!r} in agent state.")
