"""Heuristic changed-symbol and import extraction.

Regex per language, not a real parser -- deliberately shallow and easily
extended (add a language = add a row). Good enough to tell the planner "these
functions/classes moved" and "these imports appeared".
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LangRules:
    symbol_res: tuple[tuple[str, re.Pattern[str]], ...]
    import_res: tuple[re.Pattern[str], ...]


def _c(p: str) -> re.Pattern[str]:
    return re.compile(p)


RULES: dict[str, LangRules] = {
    "Python": LangRules(
        symbol_res=(
            ("function", _c(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)")),
            ("class", _c(r"^\s*class\s+([A-Za-z_]\w*)")),
        ),
        import_res=(
            _c(r"^\s*import\s+([A-Za-z_][\w.]*)"),
            _c(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import"),
        ),
    ),
    "JavaScript": LangRules(
        symbol_res=(
            ("function", _c(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)")),
            ("function", _c(r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(")),
            ("class", _c(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)")),
        ),
        import_res=(
            _c(r"""^\s*import\s+.*?from\s+['"]([^'"]+)['"]"""),
            _c(r"""require\(\s*['"]([^'"]+)['"]\s*\)"""),
        ),
    ),
    "Go": LangRules(
        symbol_res=(
            ("function", _c(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)")),
            ("class", _c(r"^\s*type\s+([A-Za-z_]\w*)\s+struct")),
        ),
        import_res=(_c(r'^\s*"([^"]+)"'),),
    ),
    "Rust": LangRules(
        symbol_res=(
            ("function", _c(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)")),
            ("class", _c(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_]\w*)")),
        ),
        import_res=(_c(r"^\s*use\s+([A-Za-z_][\w:]*)"),),
    ),
    "Java": LangRules(
        symbol_res=(
            (
                "class",
                _c(r"^\s*(?:public|private|protected)?\s*(?:final\s+)?class\s+([A-Za-z_]\w*)"),
            ),
            (
                "function",
                _c(r"^\s*(?:public|private|protected)[\w<>\[\], ]+\s+([A-Za-z_]\w*)\s*\("),
            ),
        ),
        import_res=(_c(r"^\s*import\s+([A-Za-z_][\w.]*)"),),
    ),
}
RULES["TypeScript"] = RULES["JavaScript"]


def extract(language: str, added: list[str], removed: list[str]) -> dict:
    rules = RULES.get(language)
    if rules is None:
        return {
            "added_symbols": [],
            "removed_symbols": [],
            "added_imports": [],
            "removed_imports": [],
        }

    def syms(lines: list[str]) -> list[tuple[str, str]]:
        out = []
        for ln in lines:
            for kind, rx in rules.symbol_res:
                m = rx.match(ln)
                if m:
                    out.append((kind, m.group(1)))
        return out

    def imps(lines: list[str]) -> list[str]:
        out = []
        for ln in lines:
            for rx in rules.import_res:
                m = rx.search(ln)
                if m:
                    out.append(m.group(1))
        return out

    return {
        "added_symbols": syms(added),
        "removed_symbols": syms(removed),
        "added_imports": sorted(set(imps(added))),
        "removed_imports": sorted(set(imps(removed))),
    }
