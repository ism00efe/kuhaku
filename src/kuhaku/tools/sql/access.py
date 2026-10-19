"""Who may see which tables, columns and rows.

The rule is the one RAG applies to documents, applied here to tables and columns:

- an **untagged** table or column is visible to everyone;
- a **tagged** one is visible only when one of its tags is among the caller's roles;
- a tagged one with **no** ``AuthContext`` is invisible.

Flat set intersection -- no hierarchy, no wildcard, exact case-sensitive strings. A
column is visible only when its table is; a table left with no visible column is
treated as invisible too (``COUNT(*)`` over it would still disclose its size).

Row filters are predicates the tool ANDs into every read of a table, per caller -- a
tenant id, an owner, a region. They are written by the application, not the model, and
are the only SQL that is ever trusted.

Whatever the caller is not entitled to does not exist as far as the model and the
validator are concerned: :func:`entitled_schema` produces the only schema either of
them ever sees.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from kuhaku.core.auth import AuthContext
from kuhaku.core.exceptions import ConfigError

from .catalog import Column, SchemaCatalog, Table, split_column_ref


@dataclass(frozen=True)
class RowFilter:
    """A predicate over one table's columns, e.g. ``RowFilter("tenant_id = :tenant",
    {"tenant": "acme"})``. Values are passed as bind parameters (``:name``), never
    spliced into the text. The predicate may use any column of the table, including
    ones the caller cannot see."""

    predicate: str
    params: Mapping[str, Any] = field(default_factory=dict)


RowFilterSource = RowFilter | Callable[[AuthContext | None], RowFilter | None]
"""A fixed filter for everyone, or a function from the caller to their filter.
Returning ``None`` means no filter for that caller; raising fails the request."""


def is_entitled(tags: Iterable[str], auth_context: AuthContext | None) -> bool:
    """The RAG rule (``kuhaku.tools.rag.retriever.is_entitled``), restated here because
    tools do not import each other. The two must always agree."""

    tags = tuple(tags)
    if not tags:
        return True
    if auth_context is None:
        return False
    return not set(tags).isdisjoint(auth_context.roles)


class AccessPolicy:
    """Table tags, column tags and row filters.

    ``tables`` maps a table name to its tags; ``columns`` maps ``"table.column"`` to its
    tags; ``row_filters`` maps a table name to a :data:`RowFilterSource`. Every name is
    checked against the catalog when the tool is built, so a misspelt name fails then
    instead of silently leaving the real table unprotected.
    """

    def __init__(
        self,
        *,
        tables: Mapping[str, Iterable[str]] | None = None,
        columns: Mapping[str, Iterable[str]] | None = None,
        row_filters: Mapping[str, RowFilterSource] | None = None,
    ) -> None:
        self.tables: Mapping[str, tuple[str, ...]] = MappingProxyType(
            {name: _tags(tags, name) for name, tags in (tables or {}).items()}
        )
        self.columns: Mapping[str, tuple[str, ...]] = MappingProxyType(
            {ref: _tags(tags, ref) for ref, tags in (columns or {}).items()}
        )
        for ref in self.columns:
            split_column_ref(ref)
        filters = dict(row_filters or {})
        for name, source in filters.items():
            if not isinstance(source, RowFilter) and not callable(source):
                raise ConfigError(
                    f"row_filters['{name}'] must be a RowFilter or a callable returning one."
                )
        self.row_filters: Mapping[str, RowFilterSource] = MappingProxyType(filters)

    def check(self, catalog: SchemaCatalog) -> None:
        """Raise :class:`ConfigError` if any name in the policy is not in ``catalog``."""

        for name in [*self.tables, *self.row_filters]:
            if name not in catalog:
                raise ConfigError(
                    f"The access policy names table '{name}', which is not in the catalog."
                )
        for ref in self.columns:
            table_name, column_name = split_column_ref(ref)
            table = catalog.get(table_name)
            if table is None or table.column(column_name) is None:
                raise ConfigError(
                    f"The access policy names column '{ref}', which is not in the catalog."
                )

    def table_tags(self, table: Table) -> tuple[str, ...]:
        return _lookup(self.tables, table.name)

    def column_tags(self, table: Table, column: Column) -> tuple[str, ...]:
        return _lookup(self.columns, f"{table.name}.{column.name}")

    def row_filter(self, table: Table) -> RowFilterSource | None:
        for name, source in self.row_filters.items():
            if name.casefold() == table.name.casefold():
                return source
        return None


def _lookup(mapping: Mapping[str, tuple[str, ...]], name: str) -> tuple[str, ...]:
    # Names resolve without regard to case everywhere else, so a policy keyed "Orders"
    # must protect the table "orders" too. Tags from every matching key are combined.
    key = name.casefold()
    found: list[str] = []
    for candidate, tags in mapping.items():
        if candidate.casefold() == key:
            found.extend(tags)
    return tuple(found)


def _tags(tags: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(tags, str):
        raise ConfigError(f"Tags for '{name}' must be a list of strings, not one string.")
    result = tuple(tags)
    for tag in result:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"Access tags for '{name}' must be non-empty strings; got {tag!r}.")
    if not result:
        raise ConfigError(
            f"Tags for '{name}' are empty. Leave the name out to make it visible to everyone."
        )
    return result


@dataclass(frozen=True)
class EntitledTable:
    """One table as a particular caller may see it. ``table.columns`` holds only the
    entitled columns; ``base_columns`` every column of the real table, for resolving the
    trusted row filter and nothing else."""

    table: Table
    base_columns: tuple[str, ...]
    row_filter: RowFilter | None = None

    @property
    def name(self) -> str:
        return self.table.name


class EntitledSchema:
    """The catalog as one caller is entitled to see it -- the only schema the model is
    shown and the only one names are resolved against."""

    def __init__(self, tables: Iterable[EntitledTable]) -> None:
        self._tables = {entry.name.casefold(): entry for entry in tables}

    @property
    def tables(self) -> tuple[EntitledTable, ...]:
        return tuple(self._tables.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self._tables.values())

    def get(self, name: str) -> EntitledTable | None:
        return self._tables.get(name.casefold())

    def subset(self, names: Iterable[str]) -> EntitledSchema:
        keys = {name.casefold() for name in names}
        return EntitledSchema(e for k, e in self._tables.items() if k in keys)

    def __len__(self) -> int:
        return len(self._tables)


def entitled_schema(
    catalog: SchemaCatalog, policy: AccessPolicy, auth_context: AuthContext | None
) -> EntitledSchema:
    entries: list[EntitledTable] = []
    for table in catalog.tables:
        if not is_entitled(policy.table_tags(table), auth_context):
            continue
        columns = tuple(
            column
            for column in table.columns
            if is_entitled(policy.column_tags(table, column), auth_context)
        )
        if not columns:
            continue
        source = policy.row_filter(table)
        row_filter = source(auth_context) if callable(source) else source
        if row_filter is not None and not isinstance(row_filter, RowFilter):
            raise ConfigError(
                f"The row filter for '{table.name}' returned {type(row_filter).__name__}, "
                f"not a RowFilter or None."
            )
        visible = Table(table.name, columns, table.description)
        if source is not None:
            # Sample values are read without any row filter applied, so for a filtered
            # table they could show one tenant another tenant's values.
            visible = Table(
                table.name,
                tuple(Column(c.name, c.type, c.description) for c in columns),
                table.description,
            )
        entries.append(
            EntitledTable(
                visible,
                tuple(column.name for column in table.columns),
                row_filter,
            )
        )
    return EntitledSchema(entries)
