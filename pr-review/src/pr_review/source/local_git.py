"""Build a PRContext from two refs in a local checkout.

Used for local dry runs (``pr-review run --base main --head HEAD``) and by the
test suite. No network, no GitHub.
"""

from __future__ import annotations

from pathlib import Path

from pr_review.config import Limits
from pr_review.models import PRContext
from pr_review.source import base as g


class LocalGitSource:
    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        base_ref: str = "main",
        head_ref: str = "HEAD",
        title: str = "",
        body: str = "",
        diff_bytes: int = 0,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.base_ref = base_ref
        self.head_ref = head_ref
        self.title = title
        self.body = body
        # 0 = use the safety cap. Never a review budget: see Limits.diff_bytes.
        self.diff_bytes = diff_bytes or Limits().diff_bytes

    def load(self) -> tuple[PRContext, Path]:
        root = g.ensure_repo(self.repo_root)
        moved = root != Path(self.repo_root).resolve()
        mb = g.merge_base(self.base_ref, self.head_ref, root)
        diff, truncated = g.collect_diff(
            mb, self.head_ref, root, diff_bytes=self.diff_bytes
        )
        paths = g.changed_paths(mb, self.head_ref, root)
        head_sha = g.try_git(["rev-parse", self.head_ref], root).strip()
        base_sha = g.try_git(["rev-parse", mb], root).strip()
        subject = self.title or g.try_git(
            ["log", "-1", "--pretty=%s", self.head_ref], root
        ).strip()
        pr = PRContext(
            title=subject,
            body=self.body
            or g.try_git(["log", "-1", "--pretty=%b", self.head_ref], root).strip(),
            base_ref=self.base_ref,
            head_ref=self.head_ref,
            base_sha=base_sha,
            head_sha=head_sha,
            diff=diff,
            changed_paths=tuple(paths),
            repo_slug=g.repo_slug(root),
            author=g.try_git(["log", "-1", "--pretty=%an", self.head_ref], root).strip(),
            extra={"diff_truncated": truncated, "repo_root_adjusted": str(root) if moved else ""},
        )
        return pr, root
