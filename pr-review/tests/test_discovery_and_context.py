from pathlib import Path

from pr_review.analyzer.deterministic import DeterministicAnalyzer
from pr_review.context.default import DefaultContextGatherer
from pr_review.discovery.base import DISCOVERERS, merge_metadata
from pr_review.models import ContextSpec, PRContext, RepoMetadata, ReviewTask


def _meta(root: Path) -> RepoMetadata:
    parts = [DISCOVERERS.create(n).discover(root) for n in ("languages", "manifests", "layout")]
    return merge_metadata(parts, root)


def test_discovery_on_tiny_repo(tiny_repo: Path):
    m = _meta(tiny_repo)
    assert m.primary_language() == "Python"
    assert "python" in m.package_managers
    assert "pyproject.toml" in m.dependency_files
    assert any("test" in t for t in m.test_paths)
    assert "ARCHITECTURE.md" in m.architecture_docs


def test_context_respects_spec(tiny_repo: Path, sample_diff: str):
    pr = PRContext("t", "", "main", "feature", "0", "1", sample_diff)
    analysis = DeterministicAnalyzer().analyze(pr, RepoMetadata())
    meta = _meta(tiny_repo)

    shallow = ReviewTask(
        axis="correctness", depth="basic", model_tier="basic",
        context_strategy="default",
        context_spec=ContextSpec(callers_usages=False, dependency_files=False, repo_tree=False),
        input_budget_bytes=60_000,
    )
    ctx = DefaultContextGatherer().gather(
        shallow, repo_root=tiny_repo, pr=pr, analysis=analysis, meta=meta
    )
    kinds = {i.kind for i in ctx.items}
    assert "diff" in kinds
    assert "tree" not in kinds
    assert "dependency" not in kinds

    deep = ReviewTask(
        axis="structure", depth="deep", model_tier="deep", context_strategy="default",
        context_spec=ContextSpec(
            dependency_files=True, architecture_docs=True, repo_tree=True
        ),
        input_budget_bytes=60_000,
    )
    ctx2 = DefaultContextGatherer().gather(
        deep, repo_root=tiny_repo, pr=pr, analysis=analysis, meta=meta
    )
    kinds2 = {i.kind for i in ctx2.items}
    assert {"dependency", "tree"} <= kinds2
