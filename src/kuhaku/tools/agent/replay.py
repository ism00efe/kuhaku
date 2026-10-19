"""Single-use enforcement for paused agent states.

A sealed :class:`~kuhaku.tools.agent.AgentState` is a bearer token for "run these
approved calls": anyone holding it and the ``state_secret`` can resume it. The seal
stops tampering; this module stops *replay*. Every resume records the state's nonce
with a :class:`ConsumedStateStore`, and a nonce already recorded is refused.

The default, :class:`InMemoryConsumedStateStore`, only sees resumes in its own process.
Workers that share a ``state_secret`` (several processes, several hosts, a restart)
must share a store too -- :class:`SQLiteConsumedStateStore` for one host, or an
implementation of the protocol over the application's own database.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ConsumedStateStore(Protocol):
    """Remembers which paused states were already resumed.

    :meth:`consume` must be an atomic set-if-absent: of any number of concurrent calls
    with one nonce, exactly one returns ``True``. ``expires_at`` (epoch seconds) is when
    the state itself stops being resumable, so the record may be dropped after it.
    """

    def consume(self, nonce: str, expires_at: float) -> bool:
        """Record ``nonce``; ``True`` if it was not recorded before, ``False`` if it was."""
        ...


class InMemoryConsumedStateStore:
    """Process-local store: correct for one process, blind to every other.

    Entries are dropped once their state has expired; ``max_entries`` additionally
    bounds memory, dropping the oldest -- a state whose nonce was dropped that way
    could be resumed again, so keep it above the number of states paused within one
    ``state_max_age``.
    """

    def __init__(self, max_entries: int = 100_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def consume(self, nonce: str, expires_at: float) -> bool:
        now = time.time()
        with self._lock:
            while self._seen:
                oldest, expiry = next(iter(self._seen.items()))
                if expiry > now and len(self._seen) < self.max_entries:
                    break
                del self._seen[oldest]
            if nonce in self._seen:
                return False
            self._seen[nonce] = expires_at
            return True


class SQLiteConsumedStateStore:
    """A store in a SQLite file, shared by every process on the host that opens it.

    The insert is ``INSERT OR IGNORE`` on a primary key, so two workers resuming the
    same state at once cannot both succeed. Expired rows are pruned as new ones arrive.
    """

    def __init__(self, path: str | Path, *, timeout: float = 10.0) -> None:
        self.path = str(path)
        self.timeout = timeout
        if self.path == ":memory:" or self.path.startswith("file:"):
            # Each consume opens its own connection, so an in-memory database would be
            # a new, empty one every time.
            raise ValueError("SQLiteConsumedStateStore needs a file path")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kuhaku_consumed_agent_states ("
                "nonce TEXT PRIMARY KEY, expires_at REAL NOT NULL)"
            )
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)

    def consume(self, nonce: str, expires_at: float) -> bool:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM kuhaku_consumed_agent_states WHERE expires_at < ?",
                (time.time(),),
            )
            cursor = conn.execute(
                "INSERT OR IGNORE INTO kuhaku_consumed_agent_states (nonce, expires_at) "
                "VALUES (?, ?)",
                (nonce, float(expires_at)),
            )
            conn.execute("COMMIT")
            return cursor.rowcount == 1
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
