"""What the database contains: tables, columns, their descriptions and sample values.

A catalog says nothing about who may see what -- that is :class:`~.access.AccessPolicy`'s
job -- so one catalog can serve every caller and the policy can change without
re-introspecting the database.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from kuhaku.core.exceptions import ConfigError

if TYPE_CHECKING:
    from .database import Database


@dataclass(frozen=True)
class Column:
    name: str
    type: str = ""
    description: str = ""
    sample_values: tuple[Any, ...] = ()
    """Shown to the model to help it write literals that match the data (``'EU'`` rather
    than ``'Europe'``). Never shown for a table that has a row filter -- see
    :class:`~.access.AccessPolicy`."""


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    description: str = ""
    _index: dict[str, Column] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_name(self.name, "table")
        if not self.columns:
            raise ConfigError(f"Table '{self.name}' has no columns.")
        index: dict[str, Column] = {}
        for column in self.columns:
            _check_name(column.name, f"column of table '{self.name}'")
            key = column.name.casefold()
            if key in index:
                raise ConfigError(
                    f"Table '{self.name}' has two columns named '{column.name}' "
                    f"(names are compared without regard to case)."
                )
            index[key] = column
        object.__setattr__(self, "_index", index)

    def column(self, name: str) -> Column | None:
        return self._index.get(name.casefold())


def _check_name(name: str, what: str) -> None:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ConfigError(f"Invalid {what} name: {name!r}.")


class SchemaCatalog:
    """An immutable set of tables. Names are looked up without regard to case, the way
    SQLite resolves them."""

    def __init__(self, tables: Iterable[Table]) -> None:
        self._tables: dict[str, Table] = {}
        for table in tables:
            key = table.name.casefold()
            if key in self._tables:
                raise ConfigError(f"Two tables are named '{table.name}'.")
            self._tables[key] = table

    @property
    def tables(self) -> tuple[Table, ...]:
        return tuple(self._tables.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(table.name for table in self._tables.values())

    def get(self, name: str) -> Table | None:
        return self._tables.get(name.casefold())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.casefold() in self._tables

    def __len__(self) -> int:
        return len(self._tables)

    def __repr__(self) -> str:
        return f"SchemaCatalog({list(self.names)})"

    def describe(
        self,
        descriptions: Mapping[str, str] | None = None,
        *,
        columns: Mapping[str, str] | None = None,
        sample_values: Mapping[str, Iterable[Any]] | None = None,
    ) -> SchemaCatalog:
        """A copy with descriptions added. ``descriptions`` is keyed by table name,
        ``columns`` and ``sample_values`` by ``"table.column"``. A key naming nothing in
        the catalog is a :class:`ConfigError` -- a typo must not pass silently."""

        tables = dict(self._tables)
        for name, text in (descriptions or {}).items():
            table = self._require(name)
            tables[table.name.casefold()] = replace(tables[table.name.casefold()],
                                                    description=text)
        for ref, text in (columns or {}).items():
            tables = self._with_column(tables, ref, description=text)
        for ref, values in (sample_values or {}).items():
            tables = self._with_column(tables, ref, sample_values=tuple(values))
        return SchemaCatalog(tables.values())

    @classmethod
    def from_database(
        cls,
        database: Database,
        *,
        tables: Iterable[str] | None = None,
        sample_values: int = 0,
    ) -> SchemaCatalog:
        """Introspect ``database``. ``tables`` limits the catalog to those names (the
        rest of the database is then unreachable through the tool). ``sample_values``
        reads up to that many distinct non-null values per text column -- off by
        default, because it reads data at build time."""

        catalog = cls(database.introspect())
        if tables is not None:
            wanted = [catalog._require(name) for name in tables]
            catalog = cls(wanted)
        if sample_values > 0:
            sampled: list[Table] = []
            for table in catalog.tables:
                columns = tuple(
                    replace(
                        column,
                        sample_values=tuple(
                            database.distinct_values(table.name, column.name, sample_values)
                        ),
                    )
                    if _is_text(column.type)
                    else column
                    for column in table.columns
                )
                sampled.append(replace(table, columns=columns))
            catalog = cls(sampled)
        return catalog

    # --- internals --------------------------------------------------------------------

    def _require(self, name: str) -> Table:
        table = self.get(name)
        if table is None:
            raise ConfigError(f"No table named '{name}' in the catalog: {list(self.names)}.")
        return table

    def _with_column(self, tables: dict[str, Table], ref: str, **changes: Any) -> dict[str, Table]:
        table_name, column_name = split_column_ref(ref)
        table = tables.get(table_name.casefold()) or self._require(table_name)
        column = table.column(column_name)
        if column is None:
            raise ConfigError(f"Table '{table.name}' has no column '{column_name}'.")
        new_columns = tuple(replace(c, **changes) if c is column else c for c in table.columns)
        return {**tables, table.name.casefold(): replace(table, columns=new_columns)}


def split_column_ref(ref: str) -> tuple[str, str]:
    """``"orders.total"`` -> ``("orders", "total")``."""

    table, sep, column = ref.partition(".")
    if not sep or not table or not column:
        raise ConfigError(f"Expected 'table.column', got {ref!r}.")
    return table, column


def _is_text(type_name: str) -> bool:
    upper = type_name.upper()
    return not upper or any(word in upper for word in ("CHAR", "TEXT", "CLOB", "STRING"))
