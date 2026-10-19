"""Session stores, batch execution and sanitize are reachable from top-level kuhaku."""

from __future__ import annotations

import kuhaku
from kuhaku.core.batch import BatchRunner, run_batch
from kuhaku.core.sanitization import sanitize
from kuhaku.core.session import InMemorySessionStore, Session, SQLiteSessionStore


def test_core_conveniences_are_exported_from_the_package():
    assert kuhaku.Session is Session
    assert kuhaku.InMemorySessionStore is InMemorySessionStore
    assert kuhaku.SQLiteSessionStore is SQLiteSessionStore
    assert kuhaku.BatchRunner is BatchRunner
    assert kuhaku.run_batch is run_batch
    assert kuhaku.sanitize is sanitize
    for name in ("Session", "InMemorySessionStore", "SQLiteSessionStore", "BatchRunner",
                 "run_batch", "sanitize"):
        assert name in kuhaku.__all__
