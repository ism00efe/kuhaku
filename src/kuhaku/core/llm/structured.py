"""Structured output: one loop shared by native and emulated modes.

``run_structured`` asks the provider for JSON, parses it, and on a parse failure feeds
the reply and the error back as a new turn, up to ``max_repair_attempts`` times. What
changes per provider is only how one turn asks for JSON:

- ``SCHEMA`` mode: the provider's native schema constraint (``_structured_complete``).
- ``JSON`` mode: the provider's JSON mode, plus the schema in the prompt -- JSON mode
  guarantees syntax, not shape.
- ``NONE``: the schema in the prompt and a plain ``chat`` call.

Only *parsing* is repaired here. Whether the parsed value satisfies the schema is not
checked: core has no JSON Schema validator, and callers usually hold a richer one (a
Pydantic model) and their own idea of what a repair prompt should say.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..models import Message
from .base import (
    LLMResponse,
    StructuredOutputError,
    StructuredOutputMode,
    StructuredResponse,
    TokenUsage,
    combine_usage,
)

_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class StructuredRequest:
    """What a provider's native structured-output turn needs to know."""

    name: str
    schema: dict[str, Any]


def schema_name(name: str) -> str:
    """``name`` reduced to what every vendor accepts for a schema/tool name
    (``[A-Za-z0-9_-]``, at most 64 characters)."""

    cleaned = _NAME_RE.sub("_", name).strip("_")[:64]
    return cleaned or "response"


def schema_instruction(json_schema: dict[str, Any]) -> str:
    return (
        "Reply with exactly one JSON value that follows this JSON Schema, and nothing "
        "else: no prose, no code fences.\n\nJSON Schema:\n"
        + json.dumps(json_schema, ensure_ascii=False, indent=2)
    )


def with_schema_instruction(
    messages: Sequence[Message], json_schema: dict[str, Any]
) -> list[Message]:
    """``messages`` with the schema instruction added after any leading system turns."""

    messages = list(messages)
    split = 0
    while split < len(messages) and messages[split].role == "system":
        split += 1
    return [*messages[:split], Message.system(schema_instruction(json_schema)), *messages[split:]]


def repair_message(problem: str) -> Message:
    return Message.user(
        f"Your previous reply could not be parsed as JSON ({problem}). "
        f"Reply again with the JSON value only."
    )


def parse_json(text: str) -> Any:
    """Parse ``text`` as JSON, tolerating a code fence or prose around one object/array.

    Raises :class:`StructuredOutputError`.
    """

    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    if not body:
        raise StructuredOutputError("the reply is empty", raw=text)
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        first_error = exc
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = body.find(opener), body.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(body[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise StructuredOutputError(
        f"the reply is not valid JSON ({first_error.msg})", raw=text
    ) from None


_DEFS_KEYS = ("$defs", "definitions")


def wrap_in_object(schema: dict[str, Any], key: str) -> dict[str, Any]:
    """``schema`` as the one required property ``key`` of an object schema -- for an
    API that only takes an object at the root.

    ``$defs``/``definitions`` move to the new root: a ``$ref`` such as
    ``#/$defs/Item`` resolves against the document root, so leaving them nested under
    ``properties.<key>`` would break every reference.
    """

    inner = {k: v for k, v in schema.items() if k not in _DEFS_KEYS}
    wrapped: dict[str, Any] = {
        "type": "object",
        "properties": {key: inner},
        "required": [key],
    }
    for defs_key in _DEFS_KEYS:
        if defs_key in schema:
            wrapped[defs_key] = schema[defs_key]
    return wrapped


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse ``text`` into a JSON *object*, tolerating a code fence or prose around it.

    The object-only sibling of :func:`parse_json`, for callers whose reply format is
    always an object: the fallback looks for the outermost ``{...}`` only. Providers
    without a native JSON mode add fences and prose often enough that rejecting them
    outright would spend a repair attempt on formatting alone.

    Raises :class:`StructuredOutputError` (``raw`` is ``text``); a tool re-raises it as
    its own parse-error type.
    """

    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            raise StructuredOutputError("the reply contains no JSON object", raw=text) from None
        try:
            parsed = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"the reply is not valid JSON ({exc.msg})", raw=text
            ) from None
    if not isinstance(parsed, dict):
        raise StructuredOutputError("the reply is JSON but not an object", raw=text)
    return parsed


def run_structured(
    *,
    mode: StructuredOutputMode,
    native_turn: Callable[[tuple[Message, ...], StructuredRequest], LLMResponse],
    chat_turn: Callable[[tuple[Message, ...]], LLMResponse],
    messages: Sequence[Message],
    json_schema: dict[str, Any],
    name: str,
    max_repair_attempts: int,
) -> StructuredResponse:
    """The shared loop. ``native_turn`` is used unless ``mode`` is ``NONE``."""

    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be >= 0")
    if not isinstance(json_schema, dict):
        raise TypeError(f"json_schema must be a dict, got {type(json_schema).__name__}.")

    native = mode is not StructuredOutputMode.NONE
    request = StructuredRequest(name=schema_name(name), schema=json_schema)
    conversation = (
        list(messages)
        if mode is StructuredOutputMode.SCHEMA
        else with_schema_instruction(messages, json_schema)
    )

    usage: TokenUsage | None = None
    last_text = ""
    last_problem = ""
    attempts = 0
    for _ in range(1 + max_repair_attempts):
        attempts += 1
        turn = tuple(conversation)
        response = native_turn(turn, request) if native else chat_turn(turn)
        usage = combine_usage(usage, response.usage)
        try:
            data = parse_json(response.text)
        except StructuredOutputError as exc:
            last_text, last_problem = response.text, str(exc)
            conversation = [
                *conversation,
                # Some APIs reject an empty assistant turn.
                Message.assistant(response.text or "(empty reply)"),
                repair_message(last_problem),
            ]
            continue
        return StructuredResponse(
            data=data,
            text=response.text,
            usage=usage,
            model=response.model,
            finish_reason=response.finish_reason,
            native=native,
            attempts=attempts,
        )
    raise StructuredOutputError(
        f"No parseable JSON after {attempts} attempt(s): {last_problem}",
        raw=last_text,
        attempts=attempts,
    )
