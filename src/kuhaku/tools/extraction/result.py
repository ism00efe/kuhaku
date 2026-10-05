"""The outcome of one extraction, kept apart from the extracted domain data.

A caller must be able to tell "the text did not contain the value" from "the model's
output was unusable" from "a business rule rejected it" without inspecting the domain
model -- so none of those outcomes is ever encoded as a value inside it.

Steps only ever append :class:`Issue` objects; the final :class:`ExtractionStatus` is
derived from them in one place (:func:`status_from_issues`), so adding or removing a
step never needs a change to how status is decided.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


class ExtractionStatus(str, Enum):
    COMPLETE = "complete"
    """Every required field was extracted and every enabled check passed."""

    NEEDS_INPUT = "needs_input"
    """Required values are absent from the text -- ask the user for them."""

    INVALID = "invalid"
    """Values were extracted but a check (evidence, a caller's validator) rejected them."""

    FAILED = "failed"
    """No usable structured output, even after the allowed repair attempts."""

    UNROUTED = "unrouted"
    """An :class:`ExtractionRouter` found no schema that fits the text. Nothing was
    extracted -- what to do with such a text is the caller's decision, not a default."""


class IssueKind(str, Enum):
    REPAIRABLE = "repairable"
    """Something the model can fix on another attempt (malformed JSON, a schema
    violation). One still present at the end means the extraction failed."""

    NEEDS_INPUT = "needs_input"
    """The information is not in the text; the model cannot fix it, only the user can."""

    INVALID = "invalid"
    """A rule was violated by a value the model did extract."""


@dataclass(frozen=True)
class Issue:
    kind: IssueKind
    code: str
    message: str
    path: str | None = None
    """JSON Pointer to the field concerned (e.g. ``/items/0/quantity``), when there is one."""


class Provenance(str, Enum):
    SOURCE = "source"
    """The model's quoted evidence was found in the source text."""

    PRIOR = "prior"
    """Carried over unchanged from an earlier turn (``prior=``); not in this text."""

    UNGROUNDED = "ungrounded"
    """A value is present but its evidence is missing or not found in the source text."""

    MISSING = "missing"
    """No value was extracted for this field."""


@dataclass(frozen=True)
class FieldReport:
    path: str
    provenance: Provenance
    evidence: str | None = None


# Precedence when several kinds are present: an extraction that failed outright is
# reported as failed even if it also had missing fields, and a rule violation outranks
# missing input because asking the user for more would not make it valid.
_STATUS_BY_KIND = (
    (IssueKind.REPAIRABLE, ExtractionStatus.FAILED),
    (IssueKind.INVALID, ExtractionStatus.INVALID),
    (IssueKind.NEEDS_INPUT, ExtractionStatus.NEEDS_INPUT),
)


def status_from_issues(issues: Iterable[Issue]) -> ExtractionStatus:
    kinds = {issue.kind for issue in issues}
    for kind, status in _STATUS_BY_KIND:
        if kind in kinds:
            return status
    return ExtractionStatus.COMPLETE


@dataclass(frozen=True)
class ExtractionResult(Generic[M]):
    """What :meth:`Extractor.extract` returns.

    ``value`` is set only when ``status`` is ``COMPLETE``. ``data`` is the model's last
    parsed output as a plain dict, kept even when it did not validate, so a caller can
    show what was understood so far or carry it into a follow-up turn. ``fields`` is
    empty unless the evidence step is enabled.
    """

    status: ExtractionStatus
    schema_name: str
    value: M | None = None
    data: dict[str, Any] | None = None
    issues: tuple[Issue, ...] = ()
    fields: dict[str, FieldReport] = field(default_factory=dict)
    attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ExtractionStatus.COMPLETE

    @property
    def missing(self) -> tuple[str, ...]:
        """JSON Pointers of the required fields the text did not provide."""

        return tuple(
            issue.path
            for issue in self.issues
            if issue.kind is IssueKind.NEEDS_INPUT and issue.path is not None
        )
