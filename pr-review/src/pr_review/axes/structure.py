from __future__ import annotations

from pr_review.axes.base import AXES, ContextHints


@AXES.register("structure")
class StructureAxis:
    name = "structure"
    max_severity = "blocker"
    title = "Structure"

    def goal(self) -> str:
        return (
            "Investigate STRUCTURAL and ARCHITECTURAL implications: problematic "
            "module relationships, inappropriate or newly-introduced dependencies, "
            "responsibilities placed in the wrong layer, interface/contract "
            "violations, dependency direction reversed between layers, new "
            "top-level areas, and both excessive fragmentation and excessive "
            "accumulation. Weigh against any architecture documentation provided."
        )

    def hints(self) -> ContextHints:
        return ContextHints(
            wants_dependency_files=True,
            wants_architecture_docs=True,
            wants_repo_tree=True,
            wants_callers_usages=True,
        )
