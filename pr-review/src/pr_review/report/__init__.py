"""Output/reporting, kept separate from the review engine.

A reporter turns a :class:`ReviewResult` into a string. The format can change
without touching detection or verification.
"""

from pr_review.report import json_report, markdown  # noqa: F401
from pr_review.report.base import REPORTERS, Reporter

__all__ = ["REPORTERS", "Reporter"]
