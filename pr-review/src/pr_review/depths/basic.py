from __future__ import annotations

from pr_review.depths.base import DEPTHS
from pr_review.models import ContextSpec


@DEPTHS.register("basic")
class BasicDepth:
    name = "basic"
    default_model_tier = "basic"

    def context_spec(self) -> ContextSpec:
        return ContextSpec(
            changed_file_body=True,
            surrounding_lines=0,
            sibling_files=False,
            callers_usages=False,
            dependency_files=False,
            architecture_docs=False,
            repo_tree=False,
            max_files=4,
            max_bytes=12_000,
        )

    def instruction(self) -> str:
        return (
            "Reason only about what is directly visible in the diff and the changed "
            "code shown. Do not speculate about code you cannot see. Keep findings "
            "tightly grounded in the provided lines."
        )
