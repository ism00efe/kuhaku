"""The outcome of one question.

Steps only append :class:`Issue` objects; the final :class:`SQLStatus` is derived from
them in one place (:func:`status_from_issues`), as in the extraction tool.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kuhaku.core.llm.base import TokenUsage


class SQLStatus(str, Enum):
    ANSWERED = "answered"
    """Valid SQL was produced -- and run, when the pipeline executes queries."""

    NO_MATCH = "no_match"
    """The question cannot be answered from the schema the caller may see. A question
    about data the caller is not entitled to ends here too, indistinguishably from one
    about data that does not exist."""

    REJECTED = "rejected"
    """The model's SQL failed validation, even after the allowed repair attempts."""

    FAILED = "failed"
    """No usable reply from the model, or the database could not run the query."""


class IssueKind(str, Enum):
    NO_MATCH = "no_match"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class Issue:
    kind: IssueKind
    code: str
    message: str


_STATUS_BY_KIND = (
    (IssueKind.REJECTED, SQLStatus.REJECTED),
    (IssueKind.FAILED, SQLStatus.FAILED),
    (IssueKind.NO_MATCH, SQLStatus.NO_MATCH),
)


def status_from_issues(issues: Iterable[Issue]) -> SQLStatus:
    kinds = {issue.kind for issue in issues}
    for kind, status in _STATUS_BY_KIND:
        if kind in kinds:
            return status
    return SQLStatus.ANSWERED


@dataclass(frozen=True)
class SQLResult:
    """What :meth:`TextToSQL.ask` returns.

    ``sql`` and ``params`` are the validated, rendered query -- what was (or would be)
    executed, never the model's raw text -- and are set only when ``status`` is
    ``ANSWERED``. ``rows`` holds at most ``max_rows`` rows; ``truncated`` says the cap
    was reached and more rows may exist. ``answer`` is set only by the optional answer
    step. ``usage`` sums the tokens of every model call this question made.
    """

    status: SQLStatus
    question: str
    sql: str | None = None
    params: tuple[Any, ...] | dict[str, Any] = ()
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    truncated: bool = False
    explanation: str = ""
    answer: str | None = None
    issues: tuple[Issue, ...] = ()
    attempts: int = 0
    usage: TokenUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status is SQLStatus.ANSWERED
