"""Turning a model's raw reply into ``{label: confidence}``, checked against the label set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kuhaku.core.llm.base import StructuredOutputError
from kuhaku.core.llm.structured import parse_json_object as _parse_json_object

from .labels import LabelSet
from .result import UndecidedReason


class OutputParseError(ValueError):
    """The reply is not usable: not the JSON object asked for, or it names labels that
    are not in the label set. ``problems`` are fed back to the model on a repair
    attempt; ``reason`` is what the result reports if no attempt fixes them."""

    def __init__(self, problems: list[str], reason: UndecidedReason) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
        self.reason = reason


@dataclass(frozen=True)
class Reply:
    scores: dict[str, float]
    """Canonical label name -> confidence, for every label the model proposed."""

    rationale: str


def parse_reply(raw: str, labels: LabelSet) -> Reply:
    """Parse ``{"labels": [{"label": ..., "confidence": ...}], "reason": ...}``.

    Every entry is checked; all problems are collected rather than stopping at the
    first, so one repair attempt can fix them all. Unknown names are never dropped
    silently -- a reply that is half valid would otherwise pass as a smaller decision
    than the model actually made.
    """

    try:
        parsed = parse_json_object(raw)
    except (ValueError, TypeError) as exc:
        raise OutputParseError([str(exc)], UndecidedReason.UNPARSEABLE) from None

    entries = parsed.get("labels")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise OutputParseError(['"labels" must be a list'], UndecidedReason.UNPARSEABLE)

    scores: dict[str, float] = {}
    format_problems: list[str] = []
    unknown: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            format_problems.append(f'labels[{index}] must be an object with "label" and "confidence"')
            continue
        name = labels.resolve(entry.get("label"))
        if name is None:
            unknown.append(repr(entry.get("label")))
            continue
        confidence = entry.get("confidence")
        if not _is_confidence(confidence):
            format_problems.append(
                f'labels[{index}] ("{name}"): "confidence" must be a number from 0 to 1'
            )
            continue
        scores[name] = max(scores.get(name, 0.0), float(confidence))

    problems = list(format_problems)
    if unknown:
        problems.append(
            f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not in the label "
            f"set; use only: {', '.join(labels.names)}"
        )
    if problems:
        # Unknown names alone are a distinct outcome: the model understood the format
        # but kept answering outside the label set.
        reason = UndecidedReason.UNPARSEABLE if format_problems else UndecidedReason.UNKNOWN_LABEL
        raise OutputParseError(problems, reason)

    rationale = parsed.get("reason")
    return Reply(scores=scores, rationale=rationale if isinstance(rationale, str) else "")


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse ``raw`` into a JSON object (:func:`kuhaku.core.llm.structured.parse_json_object`).
    Raises ``ValueError``."""

    try:
        return _parse_json_object(raw)
    except StructuredOutputError as exc:
        raise ValueError(str(exc)) from None


def _is_confidence(value: Any) -> bool:
    # bool is an int subclass; "true" is not a confidence.
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )
