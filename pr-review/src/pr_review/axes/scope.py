from __future__ import annotations

from pr_review.axes.base import AXES, ContextHints


@AXES.register("scope")
class ScopeAxis:
    name = "scope"
    # Process judgement, not a defect: cannot fail a build on its own,
    # and speaks about the pull request rather than one file.
    max_severity = "warning"
    whole_pr = True
    title = "Scope"

    def goal(self) -> str:
        return (
            "Investigate whether the implementation MATCHES THE STATED PURPOSE of "
            "the PR (title and body): required work that appears missing, changes "
            "unrelated to the stated goal, changes far larger than the goal needs, "
            "or a mismatch between what the PR claims and what the diff does. Use "
            "the changed-file list and PR text as primary evidence."
        )

    def hints(self) -> ContextHints:
        return ContextHints(wants_repo_tree=True, wants_architecture_docs=True)
