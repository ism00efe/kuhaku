"""Build a PRContext from a GitHub Actions ``pull_request`` event.

Reads ``$GITHUB_EVENT_PATH`` for PR metadata and uses local git (the workflow
checks out with ``fetch-depth: 0``) for the diff. Falls back to
``LocalGitSource`` semantics when no event file is present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pr_review.errors import SourceError
from pr_review.models import PRContext
from pr_review.source import base as g


class GitHubActionsSource:
    def __init__(
        self,
        *,
        event_path: str | None = None,
        repo_root: str | Path = ".",
        diff_bytes: int = 16_000,
    ) -> None:
        self.event_path = event_path or os.environ.get("GITHUB_EVENT_PATH")
        self.repo_root = Path(repo_root)
        self.diff_bytes = diff_bytes

    def load(self) -> tuple[PRContext, Path]:
        if not self.event_path or not Path(self.event_path).is_file():
            raise SourceError(
                "no GitHub event file; use --base/--head for a local run instead"
            )
        event = json.loads(Path(self.event_path).read_text("utf-8"))
        pr = event.get("pull_request")
        if not pr:
            raise SourceError("event payload has no pull_request")

        root = g.ensure_repo(self.repo_root)
        moved = root != Path(self.repo_root).resolve()
        base_ref = pr["base"]["ref"]
        head_ref = pr["head"]["ref"]
        base_sha = pr["base"]["sha"]
        head_sha = pr["head"]["sha"]

        # Prefer the exact SHAs; fall back to origin/<ref>.
        base_point = base_sha
        if not g.try_git(["cat-file", "-e", base_sha], root, default="ok"):
            base_point = f"origin/{base_ref}"
        mb = g.merge_base(base_point, head_sha or head_ref, root)
        diff, truncated = g.collect_diff(mb, head_sha or "HEAD", root, diff_bytes=self.diff_bytes)
        paths = [f["filename"] for f in _files_from_event(event)] or g.changed_paths(
            mb, head_sha or "HEAD", root
        )

        ctx = PRContext(
            title=pr.get("title", ""),
            body=pr.get("body") or "",
            base_ref=base_ref,
            head_ref=head_ref,
            base_sha=base_sha,
            head_sha=head_sha,
            diff=diff,
            changed_paths=tuple(paths),
            repo_slug=(event.get("repository") or {}).get("full_name", ""),
            number=pr.get("number"),
            author=(pr.get("user") or {}).get("login", ""),
            extra={"diff_truncated": truncated, "repo_root_adjusted": str(root) if moved else ""},
        )
        return ctx, root


def _files_from_event(event: dict) -> list[dict]:
    # GitHub does not embed the file list in the pull_request event; kept as a
    # hook for sources that do (e.g. a future gh-api source).
    return event.get("pull_request", {}).get("_files", []) or []
