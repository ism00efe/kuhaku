from __future__ import annotations

from pr_review.axes.base import AXES, ContextHints


@AXES.register("correctness")
class CorrectnessAxis:
    name = "correctness"
    max_severity = "blocker"
    title = "Correctness"

    def goal(self) -> str:
        return (
            "Investigate CORRECTNESS of the changed code: logical bugs, incorrect "
            "behaviour, unhandled edge cases, missing or wrong error handling, "
            "resource leaks, broken assumptions, invalid state transitions, "
            "off-by-one and boundary errors, and incorrect handling of null/empty "
            "or concurrent access that is visible in the change."
        )

    def hints(self) -> ContextHints:
        return ContextHints(wants_callers_usages=True, wants_sibling_files=True)
