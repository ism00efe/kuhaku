"""The outcome of one classification.

A classifier that cannot decide says so: an ``UNDECIDED`` result carries no label and a
machine-readable :class:`UndecidedReason`. Nothing ever falls back to a default label --
which way an unclassifiable text should go (a human queue, a catch-all intent) is a
domain decision, so it belongs to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kuhaku.core.llm.base import TokenUsage


class ClassificationStatus(str, Enum):
    DECIDED = "decided"
    """At least one label was chosen (exactly one in single-label mode)."""

    UNDECIDED = "undecided"
    """No label was chosen. ``undecided`` on the result says why."""


class UndecidedReason(str, Enum):
    NO_LABEL = "no_label"
    """The model answered that no label fits the text."""

    LOW_CONFIDENCE = "low_confidence"
    """Labels were proposed, but none reached ``min_confidence``."""

    AMBIGUOUS = "ambiguous"
    """Single-label mode: two labels tied for the top confidence, or rules matched
    different labels and there was no model stage to settle it."""

    UNKNOWN_LABEL = "unknown_label"
    """The model kept naming labels outside the label set, after the allowed repairs."""

    UNPARSEABLE = "unparseable"
    """The model's reply was not usable JSON, after the allowed repairs."""

    REJECTED = "rejected"
    """A decision was made, then vetoed by a later step (a caller's validator)."""


@dataclass(frozen=True)
class ClassificationResult:
    """What :meth:`Classifier.classify` returns.

    ``labels`` is empty unless ``status`` is ``DECIDED``. ``scores`` holds the
    confidence of every label the deciding stage proposed -- including ones below the
    threshold, so a caller can see what an ``UNDECIDED`` result almost was; a rule
    match scores ``1.0``. ``decided_by`` names the stage that produced the outcome
    (``"rules"``, ``"classify"``, ``"validate"``, ...) and ``reason`` says why in words:
    the rule that matched, the model's rationale, or why it abstained.

    ``usage`` sums the token counts of every model call this classification made,
    repairs and fallbacks included; ``None`` when no call was made or the provider
    reports no usage. ``trace_id`` is the id every log record and span of this call
    carries.
    """

    status: ClassificationStatus
    labels: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    decided_by: str = ""
    reason: str = ""
    undecided: UndecidedReason | None = None
    attempts: int = 0
    usage: TokenUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ClassificationStatus.DECIDED

    @property
    def label(self) -> str | None:
        """The chosen label in single-label mode; the highest-scoring one in multi-label
        mode; ``None`` when undecided."""

        return self.labels[0] if self.labels else None


def add_usage(total: TokenUsage | None, call: TokenUsage | None) -> TokenUsage | None:
    """``total + call``, where a count either side leaves unreported stays unreported
    only if neither side has it."""

    if call is None:
        return total
    if total is None:
        return call
    return TokenUsage(
        prompt_tokens=_add(total.prompt_tokens, call.prompt_tokens),
        completion_tokens=_add(total.completion_tokens, call.completion_tokens),
    )


def _add(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b
