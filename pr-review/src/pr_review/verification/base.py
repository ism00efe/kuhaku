"""Verifier contract."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pr_review.models import (
    DeterministicAnalysis,
    Finding,
    PRContext,
    RepoMetadata,
    VerdictResult,
)
from pr_review.registry import Registry


@runtime_checkable
class Verifier(Protocol):
    name: str

    def verify(
        self,
        finding: Finding,
        *,
        repo_root: Path,
        pr: PRContext,
        analysis: DeterministicAnalysis,
        meta: RepoMetadata,
        prior: VerdictResult | None = None,
    ) -> VerdictResult:
        """Judge ``finding``.

        ``prior`` carries what an earlier (cheaper) verifier already
        established, so a later one is not asked to re-decide blind. Ignoring
        it is allowed; a verifier that runs first simply receives ``None``.
        """
        ...


VERIFIERS: Registry[Verifier] = Registry("verifier")
