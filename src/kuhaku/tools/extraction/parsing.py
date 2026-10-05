"""Turning a model's raw reply into the response envelope, and pydantic errors into issues."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from .result import Issue, IssueKind

# Codes of the issues schema validation produces. A step that re-validates ``ctx.data``
# (continuation, after merging earlier turns in) drops these first so the same problem
# is never reported twice.
SCHEMA_ISSUE_CODES = frozenset({"missing", "schema_error"})


class OutputParseError(ValueError):
    """The reply is not the JSON object that was asked for."""


def parse_envelope(raw: str) -> dict[str, Any]:
    """Parse ``raw`` into ``{"data": {...}, ...}``."""

    parsed = parse_json_object(raw)
    if not isinstance(parsed.get("data"), dict):
        raise OutputParseError('the reply has no "data" object')
    return parsed


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse ``raw`` into a JSON object.

    Tolerates a code fence or stray prose around the object even though the prompt
    forbids both: providers without native JSON mode add them often enough that
    rejecting outright would spend a repair attempt on formatting alone.
    """

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise OutputParseError("the reply contains no JSON object") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OutputParseError(f"the reply is not valid JSON ({exc.msg})") from None

    if not isinstance(parsed, dict):
        raise OutputParseError("the reply is JSON but not an object")
    return parsed


def validate_data(
    model: type[BaseModel], data: dict[str, Any]
) -> tuple[BaseModel | None, list[Issue], list[Issue]]:
    """Validate ``data`` against ``model``: ``(instance or None, missing, other errors)``."""

    try:
        return model.model_validate(data), [], []
    except ValidationError as exc:
        missing, errors = issues_from_validation_error(exc)
        return None, missing, errors


def json_pointer(loc: Sequence[Any]) -> str:
    """``("items", 0, "name")`` -> ``/items/0/name`` (RFC 6901 escaping)."""

    if not loc:
        return ""
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in loc)


def issues_from_validation_error(exc: ValidationError) -> tuple[list[Issue], list[Issue]]:
    """Split pydantic errors into (missing values, everything else).

    A value is "missing" when the key is absent or the model answered null for a field
    that does not accept null -- the prompt tells it to use null for values the text
    does not contain, so a repair attempt cannot conjure them up. Every other error is
    something the model got wrong and may fix.
    """

    missing: list[Issue] = []
    errors: list[Issue] = []
    for error in exc.errors():
        path = json_pointer(error["loc"])
        if error["type"] == "missing" or error.get("input", ...) is None:
            missing.append(
                Issue(IssueKind.NEEDS_INPUT, "missing", f"{path or '/'}: not in the text", path)
            )
        else:
            errors.append(
                Issue(IssueKind.REPAIRABLE, "schema_error", f"{path or '/'}: {error['msg']}", path)
            )
    return missing, errors
