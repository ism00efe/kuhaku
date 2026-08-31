"""Default context strategy.

Honours ``task.context_spec`` (the depth's budget, already narrowed by the
axis). Uses cheap, language-agnostic techniques: file slicing, sibling listing,
substring "usage" search, dependency-file inclusion, architecture docs. Not a
call graph -- that is a future strategy behind the same interface.
"""

from __future__ import annotations

import os
from pathlib import Path

from pr_review.context.base import CONTEXT_STRATEGIES
from pr_review.diff import parse_diff, render
from pr_review.models import (
    ContextItem,
    DeterministicAnalysis,
    PRContext,
    RepoMetadata,
    ReviewContext,
    ReviewTask,
)

_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target"}


@CONTEXT_STRATEGIES.register("default")
class DefaultContextGatherer:
    name = "default"
    config_max_listed = 60

    def gather(
        self,
        task: ReviewTask,
        *,
        repo_root: Path,
        pr: PRContext,
        analysis: DeterministicAnalysis,
        meta: RepoMetadata,
    ) -> ReviewContext:
        spec = task.context_spec
        ctx = ReviewContext(task=task)
        budget = task.input_budget_bytes
        used = 0

        def add(kind: str, label: str, content: str) -> None:
            nonlocal used
            if not content or used >= budget:
                return
            room = budget - used
            body = content if len(content) <= room else content[:room] + "\n… (truncated)"
            ctx.items.append(ContextItem(kind=kind, label=label, content=body))
            used += len(body)

        # The files this pass is responsible for. Empty means the whole change.
        scope = list(task.files) if task.files else [f.path for f in analysis.files]
        in_scope = set(scope)

        # 1. The patch, restricted to this pass. Added first and never dropped:
        #    everything else is supporting material for it.
        by_path = {fd.path: fd for fd in parse_diff(pr.diff)}
        patch = "\n".join(render(by_path[p]) for p in scope if p in by_path)
        add("diff", f"unified diff ({pr.base_ref}...{pr.head_ref})", patch)

        # 2. What this pass does NOT cover. Cheap, and it stops the model
        #    reasoning as though the slice it was handed were the whole change.
        elsewhere = [f for f in analysis.files if f.path not in in_scope]
        if elsewhere:
            listed = elsewhere[: self.config_max_listed]
            rows = [
                f"  {f.status:8} +{f.additions} -{f.deletions} {f.path}" for f in listed
            ]
            if len(elsewhere) > len(listed):
                rows.append(f"  … and {len(elsewhere) - len(listed)} more")
            add(
                "note",
                f"also changed by this PR, reviewed in another pass ({len(elsewhere)} files)",
                "\n".join(rows),
            )

        changed = [p for p in scope if p not in set(analysis.deleted_paths)]

        # 3. Bodies of the files this pass owns.
        if spec.changed_file_body:
            for path in changed:
                text = _read(repo_root / path)
                if text is None:
                    continue
                if spec.surrounding_lines and analysis:
                    text = _windows(text, path, analysis, spec.surrounding_lines)
                add("file", path, text)

        # 4. Supporting context, most informative first, so a tight budget
        #    sheds the least useful material rather than whatever came last.
        if spec.callers_usages:
            names = sorted({s.name for s in analysis.changed_symbols if len(s.name) > 2})[:8]
            for hit in self._usages(repo_root, names, changed, limit=40):
                add("usage", hit[0], hit[1])

        if spec.sibling_files:
            for sib in self._siblings(repo_root, changed, limit=spec.max_files):
                add("file", f"{sib} (sibling)", _read(repo_root / sib) or "")

        if spec.dependency_files:
            for dep in meta.dependency_files[:6]:
                add("dependency", dep, _read(repo_root / dep) or "")

        if spec.architecture_docs:
            for doc in (meta.architecture_docs or [])[:3]:
                add("doc", doc, _read(repo_root / doc) or "")

        if spec.repo_tree:
            add("tree", "repository layout", _tree(repo_root))

        return ctx

    # ------------------------------------------------------------------ #

    def _siblings(self, root: Path, changed: list[str], *, limit: int) -> list[str]:
        seen = set(changed)
        out: list[str] = []
        for path in changed:
            d = (root / path).parent
            if not d.is_dir():
                continue
            for entry in sorted(d.iterdir()):
                rel = entry.relative_to(root).as_posix()
                if entry.is_file() and rel not in seen and entry.suffix == Path(path).suffix:
                    out.append(rel)
                    seen.add(rel)
                if len(out) >= limit:
                    return out
        return out

    def _usages(
        self, root: Path, names: list[str], changed: list[str], *, limit: int
    ) -> list[tuple[str, str]]:
        if not names:
            return []
        changed_set = set(changed)
        hits: list[tuple[str, str]] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                fp = Path(dirpath) / fn
                rel = fp.relative_to(root).as_posix()
                if rel in changed_set or fp.suffix not in _CODE_SUFFIXES:
                    continue
                text = _read(fp, max_bytes=120_000)
                if text is None:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if any(n in line for n in names):
                        lo = max(0, i - 3)
                        snippet = "\n".join(text.splitlines()[lo : i + 2])
                        hits.append((f"{rel}:{i}", snippet))
                        break
                if len(hits) >= limit:
                    return hits
        return hits


_CODE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".rb",
    ".php", ".cs", ".c", ".h", ".cc", ".cpp", ".swift", ".scala",
}


def _read(path: Path, *, max_bytes: int = 60_000) -> str | None:
    try:
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _windows(
    text: str, path: str, analysis: DeterministicAnalysis, radius: int
) -> str:
    lines = text.splitlines()
    names = [s.name for s in analysis.changed_symbols if s.path == path]
    if not names:
        return text if len(lines) <= 400 else "\n".join(lines[:400]) + "\n… (truncated)"
    keep: set[int] = set()
    for i, ln in enumerate(lines):
        if any(n in ln for n in names):
            keep.update(range(max(0, i - radius), min(len(lines), i + radius)))
    if not keep:
        return "\n".join(lines[:400])
    out, prev = [], -2
    for i in sorted(keep):
        if i != prev + 1:
            out.append(f"… (line {i + 1})")
        out.append(lines[i])
        prev = i
    return "\n".join(out)


def _tree(root: Path, *, max_entries: int = 200) -> str:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
        )
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth > 3:
            dirnames[:] = []
            continue
        indent = "  " * depth
        if str(rel) != ".":
            out.append(f"{indent}{rel.name}/")
        for fn in sorted(filenames)[:12]:
            out.append(f"{indent}  {fn}")
        if len(out) >= max_entries:
            out.append("… (truncated)")
            break
    return "\n".join(out)
