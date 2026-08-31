"""Detect repository layout: top-level dirs, config, tests, docs, CI, linters."""

from __future__ import annotations

from pathlib import Path

from pr_review.discovery.base import DISCOVERERS
from pr_review.models import RepoMetadata

_CONFIG_FILES = {
    ".editorconfig", ".gitignore", ".pre-commit-config.yaml", "tox.ini", "noxfile.py",
    ".flake8", ".pylintrc", "ruff.toml", ".ruff.toml", "mypy.ini", ".golangci.yml",
    ".golangci.yaml", ".eslintrc", ".eslintrc.js", ".eslintrc.json", ".prettierrc",
    "rustfmt.toml", ".rubocop.yml", "Makefile", "justfile", ".env.example",
}
_LINTER_MARKERS = {
    "ruff.toml": "ruff", ".ruff.toml": "ruff", ".flake8": "flake8",
    ".pylintrc": "pylint", "mypy.ini": "mypy", ".golangci.yml": "golangci-lint",
    ".golangci.yaml": "golangci-lint", ".eslintrc": "eslint", ".eslintrc.js": "eslint",
    ".eslintrc.json": "eslint", ".rubocop.yml": "rubocop",
}
_TEST_DIR_NAMES = {"tests", "test", "spec", "__tests__", "e2e", "it"}
_DOC_DIR_NAMES = {"docs", "doc", "documentation"}
_ARCH_DOC_NAMES = {
    "architecture.md", "architecture.rst", "design.md", "adr", "docs/architecture.md",
    "contributing.md", "agents.md",
}


@DISCOVERERS.register("layout")
class LayoutDiscoverer:
    name = "layout"

    def discover(self, root: Path) -> RepoMetadata:
        meta = RepoMetadata(root=str(root))

        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                meta.top_level_dirs.append(child.name)
                low = child.name.lower()
                if low in _TEST_DIR_NAMES:
                    meta.test_paths.append(child.name + "/")
                if low in _DOC_DIR_NAMES:
                    meta.doc_paths.append(child.name + "/")
            elif child.is_file():
                if child.name in _CONFIG_FILES:
                    meta.config_files.append(child.name)
                if child.name in _LINTER_MARKERS:
                    meta.static_analysis_tools.append(_LINTER_MARKERS[child.name])
                if child.name.lower() in _ARCH_DOC_NAMES:
                    meta.architecture_docs.append(child.name)

        # Nested test dirs (e.g. src/pkg/tests) and pyproject-declared linters.
        for sub in root.rglob("*"):
            if sub.is_dir() and sub.name.lower() in _TEST_DIR_NAMES:
                rel = sub.relative_to(root).as_posix() + "/"
                if rel not in meta.test_paths and rel.count("/") <= 4:
                    meta.test_paths.append(rel)

        ci = root / ".github" / "workflows"
        if ci.is_dir():
            meta.conventions["ci"] = "github-actions"
            meta.config_files.extend(
                f".github/workflows/{p.name}" for p in sorted(ci.glob("*.y*ml"))
            )

        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text("utf-8", errors="ignore")
            for tool in ("ruff", "mypy", "black", "isort", "pytest", "flake8"):
                if f"[tool.{tool}" in text and tool not in meta.static_analysis_tools:
                    meta.static_analysis_tools.append(tool)
            if "src/" in text or 'package-dir = {"" = "src"}' in text:
                meta.conventions["source_layout"] = "src"

        meta.architecture_docs = sorted(set(meta.architecture_docs))
        return meta
