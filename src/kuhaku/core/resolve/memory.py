"""Project-scoped decision memory: JSON at ``.kuhaku/decisions.json``.

Memory stores *selections*, never consent -- an approved install is approved for that
one action, not for the future, so the §5 consent flow runs again whenever the action is
needed. And persistence is a convenience, never a dependency: an unwritable working
directory is caught, announced once, and the process continues unpersisted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from .environment import FIELDS_CONSUMED, Fingerprint, project

_SCHEMA = 1


def default_project_dir() -> Path:
    """Where ``.kuhaku/`` lives: ``KUHAKU_PROJECT_DIR`` if set, otherwise the current
    working directory. kuhaku never assumes this is writable -- :class:`JsonMemory`
    degrades to unpersisted when it is not."""

    return Path(os.environ.get("KUHAKU_PROJECT_DIR") or os.getcwd())


class Memory(Protocol):
    def get(self, kind: str, fp: Fingerprint) -> str | None: ...

    def put(self, kind: str, fp: Fingerprint, candidate_id: str) -> None: ...

    def reset(self, kind: str | None = None) -> None: ...


class JsonMemory:
    """A ``.kuhaku/decisions.json`` under a project directory.

    An unknown or newer ``schema`` value means the whole file is ignored and the
    decisions are made again -- never a crash, never a silent migration.
    """

    def __init__(self, project_dir: str | Path | None = None, *, ui=None) -> None:
        root = Path(project_dir) if project_dir is not None else default_project_dir()
        self._path = root / ".kuhaku" / "decisions.json"
        self._ui = ui
        self._degraded = False

    # -- port ---------------------------------------------------------------
    def get(self, kind: str, fp: Fingerprint) -> str | None:
        data = self._load()
        if data is None:
            return None
        entry = data.get("decisions", {}).get(kind)
        if not isinstance(entry, dict):
            return None
        consumed = FIELDS_CONSUMED.get(kind, frozenset())
        if entry.get("fingerprint") != project(fp, consumed):
            return None
        candidate = entry.get("candidate")
        return candidate if isinstance(candidate, str) else None

    def put(self, kind: str, fp: Fingerprint, candidate_id: str) -> None:
        data = self._load() or {"schema": _SCHEMA, "decisions": {}}
        data.setdefault("decisions", {})[kind] = {
            "candidate": candidate_id,
            "fingerprint": project(fp, FIELDS_CONSUMED.get(kind, frozenset())),
        }
        self._write(data)

    def reset(self, kind: str | None = None) -> None:
        if kind is None:
            try:
                self._path.unlink()
            except OSError:
                pass
            return
        data = self._load()
        if data and kind in data.get("decisions", {}):
            del data["decisions"][kind]
            self._write(data)

    # -- internals --------------------------------------------------------
    def _load(self) -> dict | None:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("schema") != _SCHEMA:
            return None
        return data

    def _write(self, data: dict) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:
            if not self._degraded:
                self._degraded = True
                if self._ui is not None:
                    self._ui.announce(
                        f"decision memory at {self._path} is not writable ({exc}); "
                        "continuing without it -- decisions will be remade next run.",
                        dedupe_key=("memory_unwritable",),
                    )
