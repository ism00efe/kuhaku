from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A minimal git repo with a base commit and a feature branch that changes code."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-b", "main"], repo)
    _git(["config", "user.email", "t@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)

    (repo / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\ndependencies = ["requests"]\n'
    )
    (repo / "src").mkdir()
    (repo / "src" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_calc.py").write_text(
        "from src.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    (repo / "ARCHITECTURE.md").write_text("# Architecture\n\nsrc/ holds the library.\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "init"], repo)

    _git(["checkout", "-b", "feature"], repo)
    (repo / "src" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\n"
        "def divide(a, b):\n    # BUG: no zero check\n    return a / b\n\n\n"
        "def modulo(a, b):\n    return a % b\n"
    )
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
        'dependencies = ["requests", "httpx"]\n'
    )
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "feat: add modulo and a new dependency"], repo)
    return repo


@pytest.fixture
def sample_diff() -> str:
    return (
        "diff --git a/src/calc.py b/src/calc.py\n"
        "index 111..222 100644\n"
        "--- a/src/calc.py\n"
        "+++ b/src/calc.py\n"
        "@@ -1,4 +1,8 @@\n"
        " def add(a, b):\n"
        "     return a + b\n"
        "+\n"
        "+\n"
        "+def modulo(a, b):\n"
        "+    return a % b\n"
        "diff --git a/pyproject.toml b/pyproject.toml\n"
        "--- a/pyproject.toml\n"
        "+++ b/pyproject.toml\n"
        "@@ -3,1 +3,1 @@\n"
        '-dependencies = ["requests"]\n'
        '+dependencies = ["requests", "httpx"]\n'
    )
