from pathlib import Path

from pr_review.models import (
    DeterministicAnalysis,
    FileChange,
    Finding,
    PRContext,
    RepoMetadata,
    Verdict,
)
from pr_review.verification.deterministic import DeterministicVerifier


def _pr():
    return PRContext("t", "", "main", "f", "0", "1", "")


def test_invalid_when_file_absent_and_not_in_diff(tmp_path: Path):
    f = Finding("correctness", "warning", "does/not/exist.py", 3, "boom", "x")
    r = DeterministicVerifier().verify(
        f, repo_root=tmp_path, pr=_pr(), analysis=DeterministicAnalysis(), meta=RepoMetadata()
    )
    assert r.verdict == Verdict.INVALID


def test_valid_when_evidence_line_present(tmp_path: Path):
    (tmp_path / "a.py").write_text("def f():\n    return 1 / 0\n")
    f = Finding("correctness", "warning", "a.py", 2, "division by zero", "return 1 / 0")
    analysis = DeterministicAnalysis(files=[FileChange("a.py", "modified")])
    r = DeterministicVerifier().verify(
        f, repo_root=tmp_path, pr=_pr(), analysis=analysis, meta=RepoMetadata()
    )
    assert r.verdict == Verdict.VALID


def test_invalid_when_line_out_of_range(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    f = Finding("correctness", "warning", "a.py", 999, "nope", "")
    analysis = DeterministicAnalysis(files=[FileChange("a.py", "modified")])
    r = DeterministicVerifier().verify(
        f, repo_root=tmp_path, pr=_pr(), analysis=analysis, meta=RepoMetadata()
    )
    assert r.verdict == Verdict.INVALID
