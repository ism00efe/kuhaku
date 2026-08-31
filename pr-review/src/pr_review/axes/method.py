from __future__ import annotations

from pr_review.axes.base import AXES, ContextHints


@AXES.register("method")
class MethodAxis:
    name = "method"
    # Process judgement, not a defect: cannot fail a build on its own,
    # and speaks about the pull request rather than one file.
    max_severity = "warning"
    whole_pr = True
    title = "Method"

    def goal(self) -> str:
        return (
            "Investigate whether the chosen IMPLEMENTATION APPROACH is appropriate: "
            "a flawed solution strategy, unnecessary complexity, inappropriate "
            "abstractions, poor algorithmic or data-structure choices, an unstated "
            "trade-off that will cause trouble later, or a materially simpler / more "
            "robust alternative that was available. Report only where the approach "
            "has a concrete gap; do not restate what was done."
        )

    def hints(self) -> ContextHints:
        return ContextHints(wants_sibling_files=True, wants_architecture_docs=True)
