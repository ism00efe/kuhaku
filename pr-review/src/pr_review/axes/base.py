"""Axis contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pr_review.registry import Registry


@dataclass(frozen=True)
class ContextHints:
    """What kinds of context this axis benefits from, if the depth allows it."""

    wants_callers_usages: bool = False
    wants_sibling_files: bool = False
    wants_dependency_files: bool = False
    wants_architecture_docs: bool = False
    wants_repo_tree: bool = False
    extra_globs: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class ReviewAxis(Protocol):
    name: str
    title: str

    def goal(self) -> str:
        """The instruction block describing what to investigate."""
        ...

    def hints(self) -> ContextHints: ...


# Two optional class attributes shape how an axis' findings are handled. They
# are read with getattr so an axis added later needs neither, and keeps today's
# behaviour if it defines neither.
#
#   max_severity: str   ceiling for this axis' findings (default "blocker")
#   whole_pr:     bool  findings judge the PR as a whole, not one file

SEVERITY_RANK = {"info": 0, "warning": 1, "blocker": 2}


def max_severity(axis: object) -> str:
    """Highest severity this axis may emit.

    An axis that judges process rather than code -- does the diff match what the
    PR claims, is the approach the right one -- must not be able to fail a build
    through ``fail-on-blocker``. Being wrong about intent is not the same kind of
    problem as a null dereference, and conflating them makes the blocker signal
    worthless.
    """
    return getattr(axis, "max_severity", "blocker")


def clamp_severity(axis: object, severity: str) -> str:
    ceiling = max_severity(axis)
    if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(ceiling, 2):
        return ceiling
    return severity


def allowed_severities(axis: object) -> list[str]:
    ceiling = SEVERITY_RANK.get(max_severity(axis), 2)
    return [name for name, rank in SEVERITY_RANK.items() if rank <= ceiling]


def is_whole_pr(axis: object) -> bool:
    return bool(getattr(axis, "whole_pr", False))


AXES: Registry[ReviewAxis] = Registry("axis")

REPORT_ONLY_PROBLEMS = (
    "Report only problems: something missing, incorrect, risky, or unclear. "
    "Do NOT report that a change is good or well designed. Do NOT summarize what "
    "the diff does. If you cannot name a concrete problem visible in the provided "
    "material, return an empty findings list -- that is the expected answer for a "
    "clean change."
)
