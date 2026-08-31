"""Reporter contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pr_review.models import ReviewResult
from pr_review.registry import Registry


@runtime_checkable
class Reporter(Protocol):
    name: str
    file_extension: str

    def render(self, result: ReviewResult) -> str: ...


REPORTERS: Registry[Reporter] = Registry("reporter")
