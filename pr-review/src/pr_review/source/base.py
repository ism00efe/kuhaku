"""PR source contract + shared git helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

from pr_review.errors import SourceError
from pr_review.models import PRContext


@runtime_checkable
class PRSource(Protocol):
    def load(self) -> tuple[PRContext, Path]:
        """Return the PR context and the local repo root to inspect."""
        ...


def run_git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SourceError("git is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise SourceError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    return out.stdout


def try_git(args: list[str], cwd: Path, default: str = "") -> str:
    try:
        return run_git(args, cwd)
    except SourceError:
        return default


def merge_base(base_ref: str, head_ref: str, cwd: Path) -> str:
    mb = try_git(["merge-base", base_ref, head_ref], cwd).strip()
    return mb or base_ref


def collect_diff(
    base: str, head: str, cwd: Path, *, diff_bytes: int
) -> tuple[str, bool]:
    """Unified diff of ``base...head``, retrying with less context if oversized."""
    for unified in (12, 5, 3):
        text = try_git(
            ["diff", f"--unified={unified}", f"{base}...{head}"], cwd
        )
        if not text:
            text = try_git(["diff", f"--unified={unified}", base, head], cwd)
        if len(text.encode()) <= diff_bytes:
            return text, False
    truncated = text.encode()[:diff_bytes].decode(errors="ignore")
    return truncated, True


def changed_paths(base: str, head: str, cwd: Path) -> list[str]:
    raw = try_git(["diff", "--name-only", f"{base}...{head}"], cwd)
    if not raw:
        raw = try_git(["diff", "--name-only", base, head], cwd)
    return [p for p in raw.splitlines() if p.strip()]


def repo_slug(cwd: Path) -> str:
    url = try_git(["config", "--get", "remote.origin.url"], cwd).strip()
    if not url:
        return ""
    url = url.removesuffix(".git")
    for sep in ("github.com/", "github.com:"):
        if sep in url:
            return url.split(sep, 1)[1]
    return ""


def ensure_repo(root: Path) -> Path:
    """Resolve ``root`` to the git top level.

    ``git diff`` emits paths relative to the repository top level regardless of
    the directory it is run from. Every later stage joins those paths onto the
    root -- to read a changed file into the review context, to check that a
    finding's cited file exists, to slice source for the verifier -- so a root
    that is a *subdirectory* makes all of it miss, silently: the review still
    runs, on the diff alone, with no file bodies and nothing to anchor findings
    to. Anchoring the root to the top level makes the two agree by construction.
    """
    root = root.resolve()
    if not (root / ".git").exists() and not try_git(["rev-parse", "--git-dir"], root):
        raise SourceError(f"{root} is not a git repository")
    top = try_git(["rev-parse", "--show-toplevel"], root).strip()
    return Path(top).resolve() if top else root
