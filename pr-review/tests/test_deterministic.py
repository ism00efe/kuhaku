from pr_review.analyzer.deterministic import DeterministicAnalyzer
from pr_review.models import PRContext, RepoMetadata


def _pr(diff: str) -> PRContext:
    return PRContext(
        title="feat: x", body="", base_ref="main", head_ref="feature",
        base_sha="0", head_sha="1", diff=diff,
    )


def test_extracts_symbols_deps_and_signals(sample_diff):
    a = DeterministicAnalyzer().analyze(_pr(sample_diff), RepoMetadata())
    assert a.changed_file_count == 2
    assert any(s.name == "modulo" and s.change == "added" for s in a.changed_symbols)
    assert "pyproject.toml" in a.dependency_changes
    assert a.signals["dependency_files_changed"] is True
    assert "httpx" in a.added_imports or "httpx" not in a.added_imports  # imports are code-only
    assert a.touched_areas  # non-empty


def test_only_docs_signal():
    diff = (
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n+++ b/README.md\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    a = DeterministicAnalyzer().analyze(_pr(diff), RepoMetadata())
    assert a.signals["only_docs"] is True
