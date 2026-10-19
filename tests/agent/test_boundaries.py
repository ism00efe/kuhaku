"""The package rules: the agent depends on core only, and nothing else depends on it."""

from __future__ import annotations

import ast
from pathlib import Path

import kuhaku

SRC = Path(kuhaku.__file__).parent


def _imports(package: Path) -> set[str]:
    found: set[str] = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
    return found


def test_agent_depends_on_core_but_never_on_another_tool():
    modules = _imports(SRC / "tools" / "agent")
    assert any(module.startswith("kuhaku.core") for module in modules)
    assert not any(module.startswith("kuhaku.tools.") for module in modules)


def test_core_and_other_tools_never_depend_on_the_agent():
    for package in (SRC / "core", SRC / "tools" / "rag", SRC / "tools" / "extraction"):
        assert not any(module.startswith("kuhaku.tools.agent") for module in _imports(package))


def test_core_never_imports_a_tool():
    # AGENTS.md's one hard rule. core/config.py used to import RAGSettings for a
    # `Settings.rag` field; tool settings are tool-owned now.
    assert not any(module.startswith("kuhaku.tools") for module in _imports(SRC / "core"))
