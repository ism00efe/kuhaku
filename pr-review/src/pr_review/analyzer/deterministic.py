"""Objective change facts, extracted without an LLM.

Consumes the unified diff (via :mod:`pr_review.diff`) plus the discovered
:class:`RepoMetadata`, and produces a :class:`DeterministicAnalysis`. Registered
so alternative/additional analyzers can be layered in via config.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from pr_review.analyzer import symbols
from pr_review.diff import parse_diff
from pr_review.discovery.languages import language_for
from pr_review.discovery.manifests import DEP_FILE_NAMES
from pr_review.models import (
    ChangedSymbol,
    DeterministicAnalysis,
    FileChange,
    PRContext,
    RepoMetadata,
)
from pr_review.registry import Registry

ANALYZERS: Registry[DeterministicAnalyzer] = Registry("deterministic_analyzer")

_INTERFACE_HINTS = ("openapi", "swagger", ".proto", "schema.", "graphql", "/api/", "interface ")


class DeterministicAnalyzer:
    name = "core"

    def analyze(self, pr: PRContext, meta: RepoMetadata) -> DeterministicAnalysis:
        result = DeterministicAnalysis()
        file_diffs = parse_diff(pr.diff)

        for fd in file_diffs:
            path = fd.path
            lang = language_for(path)
            is_dep = PurePosixPath(path).name in DEP_FILE_NAMES
            fc = FileChange(
                path=path,
                status=fd.status,
                additions=fd.additions,
                deletions=fd.deletions,
                old_path=fd.old_path if fd.status == "renamed" else None,
                is_dependency_manifest=is_dep,
                language=lang,
            )
            result.files.append(fc)
            result.total_additions += fd.additions
            result.total_deletions += fd.deletions
            if fd.status == "added":
                result.added_paths.append(path)
            elif fd.status == "deleted":
                result.deleted_paths.append(path)
            if is_dep:
                result.dependency_changes.append(path)

            added = [t for _, t in fd.added_lines()]
            removed = [t for _, t in fd.removed_lines()]
            sym = symbols.extract(lang, added, removed)
            added_set = {n for _, n in sym["added_symbols"]}
            removed_set = {n for _, n in sym["removed_symbols"]}
            for kind, nm in sym["added_symbols"]:
                change = "modified" if nm in removed_set else "added"
                result.changed_symbols.append(ChangedSymbol(path, nm, kind, change))
            for kind, nm in sym["removed_symbols"]:
                if nm not in added_set:
                    result.changed_symbols.append(ChangedSymbol(path, nm, kind, "removed"))
            result.added_imports.extend(sym["added_imports"])
            result.removed_imports.extend(sym["removed_imports"])

            low = path.lower()
            if any(h in low for h in _INTERFACE_HINTS) or any(
                h in ln.lower() for ln in added + removed for h in _INTERFACE_HINTS
            ):
                result.interface_changes.append(path)

        result.added_imports = sorted(set(result.added_imports))
        result.removed_imports = sorted(set(result.removed_imports))
        result.interface_changes = sorted(set(result.interface_changes))
        result.touched_areas = _areas([f.path for f in result.files])

        test_paths = tuple(meta.test_paths) or ("test", "tests", "spec", "__tests__")
        touches_tests = any(
            any(seg in f.path.lower() for seg in ("test", "spec")) for f in result.files
        )
        non_trivial = result.changed_file_count > 3 or (
            result.total_additions + result.total_deletions
        ) > 120
        result.signals = {
            "dependency_files_changed": bool(result.dependency_changes),
            "new_top_level_dir": _new_top_dirs(result, meta),
            "interface_changed": bool(result.interface_changes),
            "touches_tests": touches_tests,
            "non_trivial": non_trivial,
            "large_change": (result.total_additions + result.total_deletions) > 400
            or result.changed_file_count > 15,
            "only_docs": result.files
            and all(
                f.path.lower().endswith((".md", ".rst", ".txt")) for f in result.files
            ),
            "test_paths_known": list(test_paths),
            "symbols_removed": [
                s.name for s in result.changed_symbols if s.change == "removed"
            ],
        }
        return result


def _areas(paths: list[str]) -> list[str]:
    areas: set[str] = set()
    for p in paths:
        parts = PurePosixPath(p).parts
        if len(parts) == 1:
            areas.add("(root)")
        else:
            areas.add("/".join(parts[: min(3, len(parts) - 1)]))
    return sorted(areas)


def _new_top_dirs(result: DeterministicAnalysis, meta: RepoMetadata) -> bool:
    existing = set(meta.top_level_dirs)
    for p in result.added_paths:
        top = PurePosixPath(p).parts[0]
        if "/" in p and top not in existing:
            return True
    return False


ANALYZERS.register("core", DeterministicAnalyzer)
