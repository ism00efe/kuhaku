"""AGENTS.md's one hard rule: ``kuhaku.core`` never imports a tool."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "kuhaku"


def _imports(package: Path) -> set[str]:
    modules: set[str] = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.add(node.module)
    return modules


def test_core_never_imports_a_tool():
    # core/config.py used to import RAGSettings for a `Settings.rag` field; tool
    # settings are tool-owned now.
    assert not any(module.startswith("kuhaku.tools") for module in _imports(SRC / "core"))
