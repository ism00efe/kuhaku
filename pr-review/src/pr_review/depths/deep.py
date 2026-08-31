from __future__ import annotations

from pr_review.depths.base import DEPTHS
from pr_review.models import ContextSpec


@DEPTHS.register("deep")
class DeepDepth:
    name = "deep"
    default_model_tier = "deep"

    def context_spec(self) -> ContextSpec:
        return ContextSpec(
            changed_file_body=True,
            surrounding_lines=80,
            sibling_files=True,
            callers_usages=True,
            dependency_files=True,
            architecture_docs=True,
            repo_tree=True,
            max_files=18,
        )

    def instruction(self) -> str:
        return (
            "Perform a repository-level assessment. Use the dependency information, "
            "architecture documentation, wider usages and repository layout "
            "provided to reason about ripple effects, structural fit and "
            "regressions beyond the immediate change. Prefer a few well-justified "
            "findings over many shallow ones."
        )
