"""PostgreSQL through psycopg 3 (``pip install "kuhaku[sql-postgres]"``).

The driver is imported when :class:`PostgresDatabase` is constructed, never at package
import, so the SQLite path costs nothing extra.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from kuhaku.core.exceptions import ConfigError

from .catalog import Column, Table
from .database import DBAPIDatabase, QueryRows
from .dialect import POSTGRES, Dialect
from .validator import ValidatedQuery


def _quote_ident(name: str, *, with_params: bool = False) -> str:
    if "\x00" in name:
        raise ConfigError("A PostgreSQL identifier cannot contain NUL.")
    quoted = '"' + name.replace('"', '""') + '"'
    # In a statement executed with parameters, psycopg reads every % as a placeholder.
    return quoted.replace("%", "%%") if with_params else quoted


class PostgresDatabase:
    """Every query runs in a ``READ ONLY`` transaction with a ``statement_timeout``
    and ``search_path`` set to ``schema`` (so unqualified names resolve exactly where
    the catalog was introspected), and is rolled back. Connect as a role that can only ``SELECT`` what the catalog
    names -- the read-only transaction stops writes, not reads of other tables through
    a validator bug, and PostgreSQL has no per-statement authorizer like SQLite's."""

    def __init__(
        self, conninfo: str, *, schema: str = "public", dialect: Dialect = POSTGRES
    ) -> None:
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                'PostgresDatabase needs psycopg: pip install "kuhaku[sql-postgres]".'
            ) from exc
        self._psycopg = psycopg
        self._conninfo = conninfo
        self._schema = schema
        self._dialect = dialect
        self._inner = DBAPIDatabase(self._connect, dialect=dialect, setup=self._setup)

    @property
    def dialect(self) -> Dialect:
        return self._dialect

    def _connect(self) -> Any:
        return self._psycopg.connect(self._conninfo)

    def _setup(self, cursor: Any, timeout_seconds: float) -> None:
        cursor.execute("SET TRANSACTION READ ONLY")
        # SET takes no bind parameters; the value is an int we computed.
        cursor.execute(f"SET LOCAL statement_timeout = {max(1, int(timeout_seconds * 1000))}")
        # The catalog was introspected from `schema`, and the rendered SQL names tables
        # unqualified: without this they would resolve through the role's own
        # search_path -- possibly to a same-named table in another schema. pg_catalog
        # and pg_temp are listed after it so neither can shadow a cataloged table.
        cursor.execute(
            f"SET LOCAL search_path TO {_quote_ident(self._schema)}, pg_catalog, pg_temp"
        )

    def introspect(self) -> list[Table]:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = %s ORDER BY table_name, ordinal_position",
                (self._schema,),
            )
            columns: dict[str, list[Column]] = defaultdict(list)
            for table, column, data_type in cursor.fetchall():
                columns[table].append(Column(column, data_type or ""))
            connection.rollback()
        finally:
            connection.close()
        return [Table(name, tuple(cols)) for name, cols in columns.items()]

    def distinct_values(self, table: str, column: str, limit: int) -> list[Any]:
        connection = self._connect()
        try:
            cursor = connection.cursor()
            quoted_column = _quote_ident(column, with_params=True)
            quoted_table = (
                f"{_quote_ident(self._schema, with_params=True)}."
                f"{_quote_ident(table, with_params=True)}"
            )
            cursor.execute(
                f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
                f"WHERE {quoted_column} IS NOT NULL LIMIT %s",
                (int(limit),),
            )
            values = [row[0] for row in cursor.fetchall()]
            connection.rollback()
            return values
        finally:
            connection.close()

    def execute(
        self, query: ValidatedQuery, *, max_rows: int, timeout_seconds: float
    ) -> QueryRows:
        return self._inner.execute(query, max_rows=max_rows, timeout_seconds=timeout_seconds)
