"""Spec §15 mechanical acceptance checks 13-15, scoped per the approved rulings.

13 -> core/resolve/** and tools/rag/resolve/**, excluding their adapters/ subpackages.
14 -> repo-wide; only the default UI implementation may name isatty.
15 -> the new packages and the modules they replace; CLI entry points are exempt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "kuhaku"

_RESOLVE_PKGS = [
    _SRC / "core" / "resolve",
    _SRC / "tools" / "rag" / "resolve",
]

# check 13, extended per amendment 7: llm_api.py also handles anthropic and vertex,
# so leaving them out would make the check weaker than it reads.
_PROVIDER_NAMES = re.compile(r"ollama|chroma|qdrant|groq|openai|anthropic|vertex", re.IGNORECASE)


def _py_files(root: Path):
    return sorted(root.rglob("*.py")) if root.exists() else []


def _rel(path: Path) -> str:
    return path.relative_to(_SRC).as_posix()


# --- Check 13 ----------------------------------------------------------------
def test_check13_no_provider_names_in_the_mechanism():
    offenders: list[str] = []
    for pkg in _RESOLVE_PKGS:
        for path in _py_files(pkg):
            if "adapters" in path.parts:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _PROVIDER_NAMES.search(line):
                    offenders.append(f"{_rel(path)}:{lineno}: {line.strip()}")
    assert not offenders, "tool-specific names leaked outside adapters/:\n" + "\n".join(offenders)


# --- Check 14 ----------------------------------------------------------------
def test_check14_isatty_only_in_default_ui():
    hits: list[str] = []
    for path in _py_files(_SRC):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "isatty" in line:
                hits.append(f"{_rel(path)}:{lineno}")
    assert hits, "expected ConsoleUI.is_interactive to use isatty"
    assert all(h.startswith("core/resolve/ui.py:") for h in hits), (
        "isatty must live only in the default UI implementation; found: " + ", ".join(hits)
    )


# --- Check 15 ----------------------------------------------------------------
_PRINT = re.compile(r"(?<![\w.])print\s*\(")


def test_check15_no_print_in_the_mechanism():
    offenders: list[str] = []
    for pkg in _RESOLVE_PKGS:
        for path in _py_files(pkg):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _PRINT.search(line):
                    offenders.append(f"{_rel(path)}:{lineno}: {line.strip()}")
    assert not offenders, "print() in library resolution code:\n" + "\n".join(offenders)


def test_check15_deleted_modules_are_actually_gone():
    """The old stderr-printing mechanism is replaced, not left beside the new one."""
    assert not (_SRC / "core" / "capabilities.py").exists()
    assert not (_SRC / "tools" / "rag" / "capabilities.py").exists()
