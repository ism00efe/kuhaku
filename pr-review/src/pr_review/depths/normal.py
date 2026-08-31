from __future__ import annotations

from pr_review.depths.base import DEPTHS
from pr_review.models import ContextSpec


@DEPTHS.register("normal")
class NormalDepth:
    name = "normal"
    default_model_tier = "normal"

    def context_spec(self) -> ContextSpec:
        return ContextSpec(
            changed_file_body=True,
            surrounding_lines=40,
            sibling_files=True,
            callers_usages=True,
            dependency_files=False,
            architecture_docs=False,
            repo_tree=False,
            max_files=10,
        )

    def instruction(self) -> str:
        return (
            "Consider the change together with its direct relationships: the "
            "callers, callees, and sibling code provided. Judge whether the change "
            "stays correct and coherent once those relationships are taken into "
            "account."
        )
