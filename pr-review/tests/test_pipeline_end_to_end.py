import json
from pathlib import Path

from pr_review.config import load_config
from pr_review.pipeline import Pipeline
from pr_review.report.base import REPORTERS
from pr_review.source.local_git import LocalGitSource


def test_full_offline_review(tiny_repo: Path):
    cfg = load_config(tiny_repo, overrides={"force_provider": "mock"})
    source = LocalGitSource(tiny_repo, base_ref="main", head_ref="feature")
    result = Pipeline(cfg).run(source)

    # discovery worked
    assert result.repo.primary_language() == "Python"
    assert "python" in result.repo.package_managers
    assert result.repo.architecture_docs

    # deterministic analysis worked
    assert result.analysis.changed_file_count >= 2
    assert "pyproject.toml" in result.analysis.dependency_changes

    # a plan was produced and tasks ran
    assert result.plan.active()
    assert result.tasks
    assert result.stats["tasks_run"] == len(result.tasks)

    # every finding carries a verification verdict
    for vf in result.findings:
        assert vf.result.method in ("deterministic", "llm", "skipped")

    # reports render
    md = REPORTERS.create("markdown").render(result)
    assert "## PR Review" in md
    payload = json.loads(REPORTERS.create("json").render(result))
    assert payload["schema"] == "pr-review/result@0.1"
    assert payload["plan"]["entries"]


def test_docs_only_change_is_shallow(tiny_repo: Path):
    (tiny_repo / "README.md").write_text("# Demo\n\nnew text\n")
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=tiny_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "docs: tweak"], cwd=tiny_repo, check=True, capture_output=True
    )
    cfg = load_config(tiny_repo, overrides={"force_provider": "mock", "classifier_enabled": False})
    result = Pipeline(cfg).run(
        LocalGitSource(tiny_repo, base_ref="HEAD~1", head_ref="HEAD")
    )
    # heuristic: docs-only -> scope only, no deep axes
    depths = {t.depth for t in result.tasks}
    assert depths <= {"basic"}
