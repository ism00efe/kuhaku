"""Discoverer contract, a shared bounded file walk, and metadata merge."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from pr_review.models import RepoMetadata
from pr_review.registry import Registry

_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", "target",
    ".idea", ".vscode", ".tox", "vendor", ".next", ".gradle", "coverage",
}


@runtime_checkable
class RepoDiscoverer(Protocol):
    name: str

    def discover(self, root: Path) -> RepoMetadata: ...


DISCOVERERS: Registry[RepoDiscoverer] = Registry("discoverer")


def walk_files(root: Path, *, max_files: int = 8000) -> Iterator[Path]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS and not d.startswith(".")]
        for name in filenames:
            yield Path(dirpath) / name
            count += 1
            if count >= max_files:
                return


def merge_metadata(parts: list[RepoMetadata], root: Path) -> RepoMetadata:
    out = RepoMetadata(root=str(root))
    for p in parts:
        for lang, n in p.languages.items():
            out.languages[lang] = out.languages.get(lang, 0) + n
        for attr in (
            "package_managers", "dependency_files", "frameworks", "top_level_dirs",
            "config_files", "test_paths", "doc_paths", "architecture_docs",
            "static_analysis_tools", "public_api_hints", "notes",
        ):
            merged = list(dict.fromkeys(getattr(out, attr) + getattr(p, attr)))
            setattr(out, attr, merged)
        out.conventions.update(p.conventions)
    return out
