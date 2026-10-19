"""Database adapters: introspection and the execution of validated queries.

The tool talks to a database only through :class:`Database`. :class:`SQLiteDatabase`
ships with the standard library and needs nothing else; :class:`DBAPIDatabase` wraps
any DB-API 2.0 connection factory; :class:`~.postgres.PostgresDatabase` (the
``kuhaku[sql-postgres]`` extra) imports its driver only when constructed.

An adapter executes only a :class:`~.validator.ValidatedQuery` -- SQL the validator
rendered -- and is the second line of defence, not the first: it opens the database
read-only where the backend can, bounds the time a query may run, the rows it may
return and the total bytes of those rows (``max_result_bytes``), and, for SQLite,
re-checks every table and column read against the validator's read set with SQLite's
own authorizer and caps the length of any one value (``max_value_bytes``) -- a
single function call building a huge string cannot be interrupted by the timeout.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tenacity import retry_if_exception, wait_exponential

from kuhaku.core.exceptions import ConfigError
from kuhaku.core.retry import call_with_retry

from .catalog import Column, Table
from .dialect import SQLITE, Dialect
from .validator import ValidatedQuery

logger = logging.getLogger(__name__)


class QueryExecutionError(RuntimeError):
    """The database refused or failed a validated query. ``timed_out`` is set when it
    was stopped for running past its time budget."""

    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


# One value may be at most this long (SQLite's SQLITE_LIMIT_LENGTH, per query); the
# whole result at most DEFAULT_MAX_RESULT_BYTES. Generous for an answer table, far
# below what hurts a server.
DEFAULT_MAX_VALUE_BYTES = 1_000_000
DEFAULT_MAX_RESULT_BYTES = 4_000_000


def _value_size(value: Any) -> int:
    if isinstance(value, str | bytes | bytearray | memoryview):
        return len(value)
    return 16


def _fetch_bounded(
    cursor: Any, max_rows: int, max_bytes: int
) -> tuple[list[tuple[Any, ...]], bool]:
    """Up to ``max_rows`` rows, stopping early once they total ``max_bytes``. The flag
    says more rows existed (or the byte budget cut the result short)."""

    rows: list[tuple[Any, ...]] = []
    total = 0
    while True:
        # Small batches, so the byte budget is checked before much is held in memory.
        wanted = min(max_rows + 1 - len(rows), 16)
        batch = cursor.fetchmany(wanted)
        for row in batch:
            if len(rows) == max_rows:
                return rows, True
            total += sum(_value_size(value) for value in row)
            if total > max_bytes:
                return rows, True
            rows.append(tuple(row))
        if len(batch) < wanted:
            return rows, False


@dataclass(frozen=True)
class QueryRows:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool
    """More rows existed than ``max_rows``; only the first ``max_rows`` are here."""


@runtime_checkable
class Database(Protocol):
    @property
    def dialect(self) -> Dialect: ...

    def introspect(self) -> list[Table]:
        """Every table the tool could be allowed to query, with its columns."""
        ...

    def distinct_values(self, table: str, column: str, limit: int) -> list[Any]:
        """Up to ``limit`` distinct non-null values of one column, for the catalog's
        sample values. Trusted code path: never reached with model SQL."""
        ...

    def execute(
        self, query: ValidatedQuery, *, max_rows: int, timeout_seconds: float
    ) -> QueryRows: ...


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class SQLiteDatabase:
    """A SQLite database file, opened read-only (``mode=ro``) for every query.

    Each query also runs under an authorizer that allows only ``SELECT``, reads of the
    tables and columns in the query's read set, and the dialect's allowed functions --
    so even a validator bug could not read a hidden column, write, ``ATTACH`` or load an
    extension. A query running past ``timeout_seconds`` is interrupted, and no value
    it produces may exceed ``max_value_bytes`` (SQLite's ``SQLITE_LIMIT_LENGTH``, set
    for the query and restored after) -- the timeout cannot stop one function call
    that builds a gigabyte string, the length limit stops it at once. The rows
    returned total at most ``max_result_bytes``.

    Views are not introspected: a query on a view is checked by the authorizer against
    the view's base tables, which the read set does not name, so a cataloged view
    could never be queried. Catalog the base tables instead.

    ``SQLiteDatabase.from_connection(conn)`` borrows an existing connection instead
    (an in-memory database, say). The authorizer and timeout still apply, but the
    connection is the caller's: it is not reopened read-only, so the authorizer is the
    only thing standing between a query and a write.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        dialect: Dialect = SQLITE,
        retry_attempts: int = 3,
        busy_timeout_seconds: float = 5.0,
        max_value_bytes: int = DEFAULT_MAX_VALUE_BYTES,
        max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
    ) -> None:
        if max_value_bytes < 1 or max_result_bytes < 1:
            raise ConfigError("max_value_bytes and max_result_bytes must be at least 1.")
        if str(path) in ("", ":memory:") or str(path).startswith("file:"):
            raise ConfigError(
                "SQLiteDatabase takes a database file path. For an in-memory database "
                "or a URI, open the connection yourself and use "
                "SQLiteDatabase.from_connection(conn)."
            )
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ConfigError(f"No SQLite database file at '{resolved}'.")
        self._uri: str | None = resolved.as_uri() + "?mode=ro"
        self._connection: sqlite3.Connection | None = None
        self._dialect = dialect
        self._retry_attempts = retry_attempts
        self._busy_timeout = busy_timeout_seconds
        self._max_value_bytes = max_value_bytes
        self._max_result_bytes = max_result_bytes

    @classmethod
    def from_connection(
        cls,
        connection: sqlite3.Connection,
        *,
        dialect: Dialect = SQLITE,
        max_value_bytes: int = DEFAULT_MAX_VALUE_BYTES,
        max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
    ) -> SQLiteDatabase:
        if max_value_bytes < 1 or max_result_bytes < 1:
            raise ConfigError("max_value_bytes and max_result_bytes must be at least 1.")
        instance = cls.__new__(cls)
        instance._uri = None
        instance._connection = connection
        instance._dialect = dialect
        instance._retry_attempts = 1
        instance._busy_timeout = 0.0
        instance._max_value_bytes = max_value_bytes
        instance._max_result_bytes = max_result_bytes
        return instance

    @property
    def dialect(self) -> Dialect:
        return self._dialect

    # --- connections ---

    def _open(self) -> tuple[sqlite3.Connection, bool]:
        if self._connection is not None:
            return self._connection, False
        assert self._uri is not None
        return sqlite3.connect(self._uri, uri=True, timeout=self._busy_timeout), True

    # --- introspection (trusted) ---

    def introspect(self) -> list[Table]:
        connection, owned = self._open()
        try:
            names = [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY name"
                )
            ]
            tables = []
            for name in names:
                info = connection.execute(f"PRAGMA table_info({_quote(name)})").fetchall()
                columns = tuple(Column(row[1], row[2] or "") for row in info)
                if columns:
                    tables.append(Table(name, columns))
            return tables
        finally:
            if owned:
                connection.close()

    def distinct_values(self, table: str, column: str, limit: int) -> list[Any]:
        connection, owned = self._open()
        try:
            sql = (
                f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
                f"WHERE {_quote(column)} IS NOT NULL LIMIT ?"
            )
            return [row[0] for row in connection.execute(sql, (int(limit),))]
        finally:
            if owned:
                connection.close()

    # --- execution ---

    def execute(
        self, query: ValidatedQuery, *, max_rows: int, timeout_seconds: float
    ) -> QueryRows:
        return call_with_retry(
            lambda: self._execute_once(query, max_rows, timeout_seconds),
            service="database",
            enabled=self._retry_attempts > 1,
            max_attempts=self._retry_attempts,
            wait=wait_exponential(multiplier=0.05, max=1.0),
            # Only lock contention is transient; every other error says something about
            # the query and would fail the same way again.
            retry=retry_if_exception(_is_locked),
        )

    def _execute_once(
        self, query: ValidatedQuery, max_rows: int, timeout_seconds: float
    ) -> QueryRows:
        connection, owned = self._open()
        deadline = time.monotonic() + timeout_seconds
        previous_length = _set_length_limit(connection, self._max_value_bytes)
        try:
            connection.set_authorizer(self._authorizer(query))
            connection.set_progress_handler(
                lambda: 1 if time.monotonic() > deadline else 0, 1000
            )
            try:
                cursor = connection.execute(query.sql, query.params)
                columns = tuple(description[0] for description in cursor.description or ())
                rows, truncated = _fetch_bounded(cursor, max_rows, self._max_result_bytes)
            except sqlite3.OperationalError as exc:
                if _is_locked(exc):
                    raise
                if "interrupted" in str(exc):
                    raise QueryExecutionError(
                        f"the query ran longer than {timeout_seconds:g}s", timed_out=True
                    ) from exc
                raise QueryExecutionError(str(exc)) from exc
            except sqlite3.Error as exc:
                raise QueryExecutionError(str(exc)) from exc
        finally:
            connection.set_authorizer(None)
            connection.set_progress_handler(None, 0)
            if previous_length is not None:
                _set_length_limit(connection, previous_length)
            if owned:
                connection.close()
        return QueryRows(columns=columns, rows=tuple(rows), truncated=truncated)

    def _authorizer(self, query: ValidatedQuery) -> Callable[..., int]:
        allowed = {
            table.casefold(): {column.casefold() for column in columns}
            for table, columns in query.read_set.items()
        }
        # LIKE is evaluated through a function of that name.
        functions = {
            name.casefold() for name in (*self._dialect.functions, *self._dialect.niladic)
        } | {"like"}

        def authorize(action: int, arg1: Any, arg2: Any, _db: Any, _trigger: Any) -> int:
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ:
                columns = allowed.get(str(arg1 or "").casefold())
                # An empty column name is a table read with no column (COUNT(*)).
                if columns is not None and (not arg2 or str(arg2).casefold() in columns):
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION:
                if str(arg2 or "").casefold() in functions:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_DENY

        return authorize


def _set_length_limit(connection: sqlite3.Connection, limit: int) -> int | None:
    """Set SQLITE_LIMIT_LENGTH; return the previous value (None where the Python build
    has no Connection.setlimit, i.e. before 3.11)."""

    setlimit = getattr(connection, "setlimit", None)
    category = getattr(sqlite3, "SQLITE_LIMIT_LENGTH", None)
    if setlimit is None or category is None:
        return None
    previous: int = setlimit(category, limit)
    return previous


def _is_locked(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc)


class DBAPIDatabase:
    """Any DB-API 2.0 driver, through a connection factory.

    ``connect`` is called once per query and the connection is closed afterwards; the
    transaction is always rolled back. ``dialect`` must carry the driver's
    ``paramstyle``. ``tables`` is the catalog this adapter reports from
    :meth:`introspect` (DB-API has no portable introspection).

    This adapter cannot make the connection read-only or bound its run time itself --
    connect with a database role that can only ``SELECT`` the cataloged tables, and
    pass ``setup(cursor, timeout_seconds)`` to run per-connection statements before
    the query: a read-only transaction and a statement timeout of
    ``timeout_seconds`` (for PostgreSQL, ``SET TRANSACTION READ ONLY`` and
    ``SET LOCAL statement_timeout``; :class:`~.postgres.PostgresDatabase` does exactly
    that). Without ``setup`` a :class:`UserWarning` is issued at construction, because
    then nothing but the role's grants bounds what a query may do or how long it runs.
    Rows returned total at most ``max_result_bytes``.
    """

    def __init__(
        self,
        connect: Callable[[], Any],
        *,
        dialect: Dialect,
        tables: Sequence[Table] = (),
        setup: Callable[[Any, float], None] | None = None,
        max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
    ) -> None:
        if max_result_bytes < 1:
            raise ConfigError("max_result_bytes must be at least 1.")
        if setup is None:
            warnings.warn(
                "DBAPIDatabase without setup=: queries run with no statement timeout and "
                "no read-only transaction. Pass setup=(cursor, timeout_seconds) -> None "
                "that sets both.",
                UserWarning,
                stacklevel=2,
            )
        self._connect = connect
        self._dialect = dialect
        self._tables = list(tables)
        self._setup = setup
        self._max_result_bytes = max_result_bytes

    @property
    def dialect(self) -> Dialect:
        return self._dialect

    def introspect(self) -> list[Table]:
        if not self._tables:
            raise ConfigError(
                "DBAPIDatabase cannot introspect a database portably; pass tables=[...]."
            )
        return list(self._tables)

    def distinct_values(self, table: str, column: str, limit: int) -> list[Any]:
        raise ConfigError("DBAPIDatabase does not read sample values; set them on the catalog.")

    def execute(
        self, query: ValidatedQuery, *, max_rows: int, timeout_seconds: float
    ) -> QueryRows:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            try:
                if self._setup is not None:
                    self._setup(cursor, timeout_seconds)
                cursor.execute(query.sql, query.params)
                columns = tuple(description[0] for description in cursor.description or ())
                rows, truncated = _fetch_bounded(cursor, max_rows, self._max_result_bytes)
            except Exception as exc:
                raise QueryExecutionError(str(exc), timed_out=_looks_like_timeout(exc)) from exc
            finally:
                connection.rollback()
        finally:
            connection.close()
        return QueryRows(columns=columns, rows=tuple(rows), truncated=truncated)


def _looks_like_timeout(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "timeout" in text or "canceling statement" in text
