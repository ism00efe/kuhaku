"""Context gatherer contract."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pr_review.models import (
    DeterministicAnalysis,
    PRContext,
    RepoMetadata,
    ReviewContext,
    ReviewTask,
)
from pr_review.registry import Registry


@runtime_checkable
class ContextGatherer(Protocol):
    name: str

    def gather(
        self,
        task: ReviewTask,
        *,
        repo_root: Path,
        pr: PRContext,
        analysis: DeterministicAnalysis,
        meta: RepoMetadata,
    ) -> ReviewContext: ...


CONTEXT_STRATEGIES: Registry[ContextGatherer] = Registry("context_strategy")
