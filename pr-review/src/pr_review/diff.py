"""A minimal unified-diff parser (no dependencies).

Handles the ``git diff`` output shape: ``diff --git`` headers, ``rename``/
``new file``/``deleted file`` markers, ``---``/``+++`` paths and ``@@`` hunks.
It is intentionally forgiving -- unknown lines are ignored rather than fatal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


@dataclass
class DiffLine:
    kind: str  # "+" | "-" | " "
    text: str
    old_lineno: int | None
    new_lineno: int | None


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class FileDiff:
    old_path: str | None
    new_path: str | None
    status: str  # "added" | "deleted" | "modified" | "renamed"
    hunks: list[Hunk] = field(default_factory=list)
    is_binary: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def additions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "+")

    @property
    def deletions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "-")

    def added_lines(self) -> list[tuple[int, str]]:
        return [
            (ln.new_lineno or 0, ln.text)
            for h in self.hunks
            for ln in h.lines
            if ln.kind == "+"
        ]

    def removed_lines(self) -> list[tuple[int, str]]:
        return [
            (ln.old_lineno or 0, ln.text)
            for h in self.hunks
            for ln in h.lines
            if ln.kind == "-"
        ]


def render(fd: FileDiff) -> str:
    """Re-emit one file's diff in unified form.

    The engine needs to hand a model *some* of the change rather than all of
    it, so the unit of selection has to be a single file's patch. Rendering
    from the parsed form -- instead of slicing the raw diff text at a byte
    offset -- guarantees every hunk handed over is syntactically whole.
    """
    head = f"diff --git a/{fd.old_path or fd.new_path} b/{fd.new_path or fd.old_path}"
    out = [head]
    if fd.status == "added":
        out.append("new file")
    elif fd.status == "deleted":
        out.append("deleted file")
    elif fd.status == "renamed":
        out.append(f"rename from {fd.old_path}")
        out.append(f"rename to {fd.new_path}")
    if fd.is_binary:
        out.append("Binary file")
        return "\n".join(out)
    out.append(f"--- a/{fd.old_path or '/dev/null'}")
    out.append(f"+++ b/{fd.new_path or '/dev/null'}")
    for h in fd.hunks:
        out.append(
            f"@@ -{h.old_start},{h.old_count} +{h.new_start},{h.new_count} @@"
            + (f" {h.section}" if h.section else "")
        )
        out.extend(f"{ln.kind}{ln.text}" for ln in h.lines)
    return "\n".join(out)


def rendered_size(fd: FileDiff) -> int:
    """Bytes :func:`render` would produce for this file."""
    return len(render(fd).encode())


def _strip_prefix(path: str) -> str | None:
    if path in ("/dev/null", ""):
        return None
    for pre in ("a/", "b/"):
        if path.startswith(pre):
            return path[len(pre) :]
    return path


def parse_diff(text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    hunk: Hunk | None = None
    old_no = new_no = 0

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("diff --git "):
            if cur is not None:
                files.append(cur)
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            op = m.group(1) if m else None
            np = m.group(2) if m else None
            cur = FileDiff(old_path=op, new_path=np, status="modified")
            hunk = None
            i += 1
            continue

        if cur is None:
            i += 1
            continue

        if line.startswith("new file mode"):
            cur.status = "added"
        elif line.startswith("deleted file mode"):
            cur.status = "deleted"
        elif line.startswith("rename from "):
            cur.old_path = line[len("rename from ") :]
            cur.status = "renamed"
        elif line.startswith("rename to "):
            cur.new_path = line[len("rename to ") :]
            cur.status = "renamed"
        elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
            cur.is_binary = True
        elif line.startswith("--- "):
            cur.old_path = _strip_prefix(line[4:]) or cur.old_path
        elif line.startswith("+++ "):
            cur.new_path = _strip_prefix(line[4:]) or cur.new_path
        elif line.startswith("@@"):
            m = _HUNK_RE.match(line)
            if m:
                old_no = int(m.group(1))
                new_no = int(m.group(3))
                hunk = Hunk(
                    old_start=old_no,
                    old_count=int(m.group(2) or 1),
                    new_start=new_no,
                    new_count=int(m.group(4) or 1),
                    section=(m.group(5) or "").strip(),
                )
                cur.hunks.append(hunk)
        elif hunk is not None and line[:1] in ("+", "-", " "):
            kind = line[0]
            body = line[1:]
            if kind == "+":
                hunk.lines.append(DiffLine("+", body, None, new_no))
                new_no += 1
            elif kind == "-":
                hunk.lines.append(DiffLine("-", body, old_no, None))
                old_no += 1
            else:
                hunk.lines.append(DiffLine(" ", body, old_no, new_no))
                old_no += 1
                new_no += 1
        elif line == r"\ No newline at end of file":
            pass

        i += 1

    if cur is not None:
        files.append(cur)

    for f in files:
        if f.status == "modified" and f.old_path is None and f.new_path:
            f.status = "added"
    return files
