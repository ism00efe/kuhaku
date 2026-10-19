"""The security core: model SQL in, the SQL that may be executed out.

Deterministic, no model involved. :meth:`Validator.validate` parses the text with the
allow-list grammar (:mod:`.parser`), resolves every table and column reference against
the caller's :class:`~.access.EntitledSchema`, and then **renders new SQL from the
resolved tree**. The model's text is never executed; what runs is built here:

- every table reference gets a fresh alias (``t1``, ``t2`` ...) and every column
  reference is written ``"alias"."column"``, every CTE gets a fresh name with a
  random per-query prefix (never derived from the catalog, so the names reveal
  nothing about which tables exist), every ``*`` is expanded into the entitled
  columns -- so how the database resolves a name can never differ from how it was
  checked here;
- a table with a row filter is replaced by ``(SELECT <entitled columns> FROM t WHERE
  <filter>)`` wherever it is read, including inside subqueries and CTEs. The filter is
  in the executed SQL only: :attr:`ValidatedQuery.display_sql` -- what the caller is
  shown -- writes ``/* row filter */ TRUE`` in its place and carries none of its
  parameters, since a filter may name hidden columns and other tenants' values;
- ``CAST`` may only name the dialect's allowed types, and function calls nest at most
  :data:`MAX_FUNCTION_DEPTH` deep with no ``REPLACE`` inside another ``REPLACE`` --
  the nesting that turns a short query into a gigabyte string;
- every operator is parenthesized, so the database evaluates exactly the parsed tree;
- string literals become bind parameters (numbers are inlined -- see :meth:`_Renderer.expr`);
- the outermost ``LIMIT`` is capped.

A name the caller is not entitled to and a name that does not exist produce the same
:class:`SQLValidationError` -- the validator is only ever given the entitled schema, so
it could not tell them apart if it tried. The error message names only what the SQL
itself said, so it is safe to show the model during repair.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from kuhaku.core.exceptions import ConfigError

from .access import EntitledSchema, EntitledTable, RowFilter
from .dialect import SQLITE, Dialect
from .parser import (
    Between,
    Binary,
    Case,
    Cast,
    ColumnRef,
    Exists,
    Expr,
    FuncCall,
    InList,
    InQuery,
    Is,
    Like,
    Literal,
    Param,
    Query,
    ScalarSubquery,
    Select,
    SQLSyntaxError,
    Star,
    SubqueryItem,
    TableItem,
    Unary,
    parse_predicate,
    parse_query,
)


class SQLValidationError(ValueError):
    """The SQL was rejected. ``code`` is stable and machine-readable; the message is
    safe to show the model. ``reference`` is the table or column name, as the SQL wrote
    it, that could not be resolved -- for the audit record, which is the one place the
    operator can see whether it named a real but non-entitled object."""

    def __init__(self, message: str, *, code: str, reference: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.reference = reference


@dataclass(frozen=True)
class ValidatedQuery:
    sql: str
    """The SQL to execute -- rendered by the validator, not the model's text."""

    params: tuple[Any, ...] | dict[str, Any]
    tables: tuple[str, ...]
    """Catalog names of every table read."""

    columns: tuple[str, ...]
    """``"table.column"`` for every entitled column read (row-filter columns excluded)."""

    output_columns: tuple[str, ...]
    limit: int
    limit_capped: bool
    """True when the model's ``LIMIT`` was absent or above the cap and the cap was used."""

    read_set: Mapping[str, frozenset[str]] = field(default_factory=dict)
    """Table -> every column the database will read, row-filter columns included -- for
    a backend that can enforce it a second time (SQLite's authorizer)."""

    display_sql: str = ""
    """``sql`` with every row-filter predicate replaced by ``/* row filter */ TRUE`` --
    what the caller may be shown. ``sql`` itself is for execution and audit only."""

    display_params: tuple[Any, ...] | dict[str, Any] = ()
    """The parameters of ``display_sql``: the query's own literals, no filter values."""


# --- scopes ---------------------------------------------------------------------------


@dataclass
class _Source:
    key: str
    alias: str
    columns: list[tuple[str, str]]  # (casefolded key, exact name)
    table: str | None = None  # catalog name, for a base table


@dataclass
class _CTEDef:
    name: str
    columns: list[tuple[str, str]]


@dataclass
class _Scope:
    parent: _Scope | None = None
    sources: list[_Source] = field(default_factory=list)
    ctes: dict[str, _CTEDef] = field(default_factory=dict)
    aliases: dict[str, Expr] = field(default_factory=dict)

    def find_cte(self, key: str) -> _CTEDef | None:
        scope: _Scope | None = self
        while scope is not None:
            if key in scope.ctes:
                return scope.ctes[key]
            scope = scope.parent
        return None


@dataclass
class _FilterScope:
    """Resolution context for a trusted row-filter predicate: one table, its real
    columns, and the filter's own parameters."""

    entry: EntitledTable
    row_filter: RowFilter
    used: set[str] = field(default_factory=set)


# Internal placeholder, replaced by the paramstyle's own form once rendering is done.
# NUL never occurs in accepted SQL or in a catalog name, so it cannot collide.
_MARK = "\x00"
_MARK_RE = re.compile("\x00(\\d+)\x00")

MAX_FUNCTION_DEPTH = 6
"""How deeply function calls may nest inside one another."""

# Functions whose output can be much longer than their input; one may not appear inside
# another's arguments. Nested REPLACE grows the string geometrically in a single
# function call, which a progress-handler timeout cannot interrupt.
_AMPLIFIERS = frozenset({"REPLACE"})


class Validator:
    """Validates and renders SQL for one caller. Cheap to build; build one per request
    from that caller's :class:`EntitledSchema`.

    ``reserved_names`` are table names the generated CTE names must avoid -- all
    real table names, entitled or not. They never influence the names chosen (those
    carry a random prefix): a collision only makes the whole query draw a new prefix,
    so the rendered names reveal nothing about the catalog. They never appear in an
    error.
    """

    def __init__(
        self,
        schema: EntitledSchema,
        *,
        dialect: Dialect = SQLITE,
        max_rows: int = 100,
        reserved_names: Iterable[str] = (),
    ) -> None:
        if max_rows < 1:
            raise ConfigError("max_rows must be at least 1.")
        self.schema = schema
        self.dialect = dialect
        self.max_rows = max_rows
        self._reserved = {name.casefold() for name in reserved_names} | {
            entry.name.casefold() for entry in schema.tables
        }

    def validate(self, sql: str) -> ValidatedQuery:
        try:
            query = parse_query(sql)
        except SQLSyntaxError as exc:
            raise SQLValidationError(str(exc), code=exc.code) from None
        while True:
            prefix = f"q{secrets.token_hex(4)}_"
            try:
                executed = _Renderer(self, prefix).render(query)
            except _NameCollision:
                continue
            # The same tree, same prefix, same counters: identical but for the filters.
            shown = _Renderer(self, prefix, display=True).render(query)
            return ValidatedQuery(
                sql=executed.sql,
                params=executed.params,
                tables=executed.tables,
                columns=executed.columns,
                output_columns=executed.output_columns,
                limit=executed.limit,
                limit_capped=executed.limit_capped,
                read_set=executed.read_set,
                display_sql=shown.sql,
                display_params=shown.params,
            )


class _NameCollision(Exception):
    """A generated CTE name equals a real table name; retry with another prefix."""


class _Renderer:
    def __init__(self, validator: Validator, prefix: str, *, display: bool = False) -> None:
        self.v = validator
        self.prefix = prefix
        self.display = display
        self.func_depth = 0
        self.amplifier_depth = 0
        self.dialect = validator.dialect
        self.values: list[Any] = []
        self.tables: dict[str, None] = {}
        self.columns: dict[str, None] = {}
        self.read_set: dict[str, set[str]] = {}
        self.alias_count = 0
        self.cte_count = 0

    # --- entry ---

    def render(self, query: Query) -> ValidatedQuery:
        text, names, limit, capped = self.query(query, None, top=True)
        sql, params = self.finish(text)
        return ValidatedQuery(
            sql=sql,
            params=params,
            tables=tuple(self.tables),
            columns=tuple(self.columns),
            output_columns=tuple(names),
            limit=limit,
            limit_capped=capped,
            read_set={table: frozenset(cols) for table, cols in self.read_set.items()},
        )

    def finish(self, text: str) -> tuple[str, tuple[Any, ...] | dict[str, Any]]:
        style = self.dialect.paramstyle
        ordered: list[Any] = []

        def placeholder(match: re.Match[str]) -> str:
            ordered.append(self.values[int(match.group(1))])
            n = len(ordered)
            return {
                "qmark": "?",
                "format": "%s",
                "numeric": f":{n}",
                "named": f":p{n}",
                "pyformat": f"%(p{n})s",
            }[style]

        sql = _MARK_RE.sub(placeholder, text)
        if style in ("named", "pyformat"):
            return sql, {f"p{i}": value for i, value in enumerate(ordered, start=1)}
        return sql, tuple(ordered)

    # --- helpers ---

    def q(self, name: str) -> str:
        quoted = '"' + name.replace('"', '""') + '"'
        return quoted.replace("%", "%%") if self.dialect.percent_escaped else quoted

    def param(self, value: Any) -> str:
        self.values.append(value)
        return f"{_MARK}{len(self.values) - 1}{_MARK}"

    def new_alias(self) -> str:
        self.alias_count += 1
        return f"t{self.alias_count}"

    def new_cte_name(self) -> str:
        self.cte_count += 1
        name = f"{self.prefix}{self.cte_count}"
        if name.casefold() in self.v._reserved:
            raise _NameCollision
        return name

    # --- queries ---

    def query(
        self, query: Query, parent: _Scope | None, *, top: bool = False
    ) -> tuple[str, list[str], int, bool]:
        scope = _Scope(parent=parent) if query.ctes else parent
        parts: list[str] = []
        if query.ctes:
            assert scope is not None
            definitions: list[str] = []
            for cte in query.ctes:
                if cte.name.key in scope.ctes:
                    raise SQLValidationError(
                        f"two CTEs are named '{cte.name.name}'", code="ambiguous"
                    )
                # Rendered with only the earlier CTEs in scope: a CTE naming itself
                # resolves to a table of that name, never to itself (no recursion).
                body, names, _, _ = self.query(cte.query, scope)
                name = self.new_cte_name()
                scope.ctes[cte.name.key] = _CTEDef(name, [(n.casefold(), n) for n in names])
                definitions.append(f"{self.q(name)} AS ({body})")
            parts.append("WITH " + ", ".join(definitions))

        rendered = [self.select(core, scope) for core in query.cores]
        names = rendered[0][1]
        for text, core_names, _ in rendered[1:]:
            if len(core_names) != len(names):
                raise SQLValidationError(
                    "every SELECT in a UNION/INTERSECT/EXCEPT must return the same number "
                    "of columns",
                    code="column_count",
                )
        body = rendered[0][0]
        for op, (text, _, _) in zip(query.operators, rendered[1:], strict=True):
            body += f" {op} {text}"
        parts.append(body)

        if query.order_by:
            core_scope = rendered[0][2] if len(rendered) == 1 else None
            terms = [self.order_term(item, names, core_scope) for item in query.order_by]
            parts.append("ORDER BY " + ", ".join(terms))

        limit = query.limit
        capped = False
        if top and (limit is None or limit > self.v.max_rows):
            limit = self.v.max_rows
            capped = True
        if limit is not None:
            parts.append(f"LIMIT {limit}")
            if query.offset is not None:
                parts.append(f"OFFSET {query.offset}")
        elif query.offset is not None:
            raise SQLValidationError("OFFSET needs a LIMIT", code="limit")
        return " ".join(parts), names, limit or 0, capped

    def order_term(self, item: Any, names: list[str], scope: _Scope | None) -> str:
        expr = item.expr
        keys = [name.casefold() for name in names]
        if isinstance(expr, Literal) and expr.kind == "number" and isinstance(expr.value, int):
            text = str(self.ordinal(expr.value, len(names)))
        elif isinstance(expr, ColumnRef) and expr.table is None and expr.column.key in keys:
            # A result-column name. Rendered as its position: an output alias that is
            # also an input column name means different things in different databases.
            text = str(keys.index(expr.column.key) + 1)
        elif scope is None:
            raise SQLValidationError(
                "ORDER BY on a UNION/INTERSECT/EXCEPT must name a result column or its "
                "position",
                code="unsupported",
            )
        else:
            text = self.expr(expr, scope)
        if item.descending is not None:
            text += " DESC" if item.descending else " ASC"
        if item.nulls:
            text += f" NULLS {item.nulls}"
        return text

    def ordinal(self, value: int, count: int) -> int:
        if not 1 <= value <= count:
            raise SQLValidationError(
                f"column position {value} is out of range (1 to {count})", code="unsupported"
            )
        return value

    def select(self, core: Select, parent: _Scope | None) -> tuple[str, list[str], _Scope]:
        scope = _Scope(parent=parent)
        from_parts: list[str] = []
        for join in core.from_:
            item_text = self.from_item(join.item, scope, parent)
            if join.kind == "FIRST":
                from_parts.append(item_text)
            elif join.kind == "COMMA":
                from_parts.append(f", {item_text}")
            else:
                keyword = {"INNER": "JOIN", "LEFT": "LEFT JOIN", "CROSS": "CROSS JOIN"}[join.kind]
                text = f" {keyword} {item_text}"
                if join.on is not None:
                    text += f" ON {self.expr(join.on, scope)}"
                from_parts.append(text)

        for column in core.columns:
            if column.alias is not None and not isinstance(column.expr, Star):
                scope.aliases.setdefault(column.alias.key, column.expr)

        select_parts: list[str] = []
        names: list[str] = []
        for column in core.columns:
            if isinstance(column.expr, Star):
                for text, name in self.expand_star(column.expr, scope):
                    select_parts.append(f"{text} AS {self.q(name)}")
                    names.append(name)
                continue
            text = self.expr(column.expr, scope)
            if column.alias is not None:
                name = column.alias.name
            elif isinstance(column.expr, ColumnRef):
                name = column.expr.column.name
            else:
                name = column.source_text
            select_parts.append(f"{text} AS {self.q(name)}")
            names.append(name)

        text = "SELECT " + ("DISTINCT " if core.distinct else "") + ", ".join(select_parts)
        if from_parts:
            text += " FROM " + "".join(from_parts)
        if core.where is not None:
            text += " WHERE " + self.expr(core.where, scope, use_aliases=True)
        if core.group_by:
            terms = []
            for expr in core.group_by:
                if isinstance(expr, Literal) and expr.kind == "number" and isinstance(
                    expr.value, int
                ):
                    terms.append(str(self.ordinal(expr.value, len(names))))
                else:
                    terms.append(self.expr(expr, scope, use_aliases=True))
            text += " GROUP BY " + ", ".join(terms)
        if core.having is not None:
            text += " HAVING " + self.expr(core.having, scope, use_aliases=True)
        return text, names, scope

    def expand_star(self, star: Star, scope: _Scope) -> list[tuple[str, str]]:
        if star.table is not None:
            sources = [s for s in scope.sources if s.key == star.table.key]
            if not sources:
                raise SQLValidationError(
                    f"unknown table '{star.table.name}'",
                    code="unknown_table",
                    reference=star.table.name,
                )
        else:
            sources = scope.sources
            if not sources:
                raise SQLValidationError("SELECT * needs a FROM clause", code="unsupported")
        expanded = []
        for source in sources:
            for _, name in source.columns:
                expanded.append((self.column_text(source, name), name))
        return expanded

    def from_item(self, item: Any, scope: _Scope, parent: _Scope | None) -> str:
        alias = self.new_alias()
        if isinstance(item, SubqueryItem):
            # No LATERAL: a FROM subquery sees the enclosing query's scope, not its
            # sibling FROM items.
            body, names, _, _ = self.query(item.query, parent)
            key = item.alias.key if item.alias else ""
            self.add_source(scope, _Source(key, alias, [(n.casefold(), n) for n in names]))
            return f"({body}) AS {self.q(alias)}"

        assert isinstance(item, TableItem)
        key = item.alias.key if item.alias else item.name.key
        cte = scope.find_cte(item.name.key)
        if cte is not None:
            self.add_source(scope, _Source(key, alias, list(cte.columns)))
            return f"{self.q(cte.name)} AS {self.q(alias)}"

        entry = self.v.schema.get(item.name.name)
        if entry is None:
            raise SQLValidationError(
                f"unknown table '{item.name.name}'",
                code="unknown_table",
                reference=item.name.name,
            )
        table = entry.table
        self.tables[table.name] = None
        reads = self.read_set.setdefault(table.name, set())
        reads.update(column.name for column in table.columns)
        columns = [(column.name.casefold(), column.name) for column in table.columns]
        self.add_source(scope, _Source(key, alias, columns, table.name))
        if entry.row_filter is None:
            return f"{self.q(table.name)} AS {self.q(alias)}"
        if self.display:
            predicate = "/* row filter */ TRUE"
        else:
            predicate = self.filter_predicate(entry, entry.row_filter)
        listed = ", ".join(self.q(column.name) for column in table.columns)
        return (
            f"(SELECT {listed} FROM {self.q(table.name)} WHERE {predicate}) "
            f"AS {self.q(alias)}"
        )

    def add_source(self, scope: _Scope, source: _Source) -> None:
        if source.key and any(existing.key == source.key for existing in scope.sources):
            raise SQLValidationError(
                f"the name '{source.key}' is used for two tables in one FROM; give them "
                f"different aliases",
                code="ambiguous",
            )
        scope.sources.append(source)

    def column_text(self, source: _Source, name: str) -> str:
        if source.table is not None:
            self.columns[f"{source.table}.{name}"] = None
        return f"{self.q(source.alias)}.{self.q(name)}"

    # --- name resolution ---

    def resolve(self, ref: ColumnRef, scope: _Scope) -> str | None:
        key = ref.column.key
        current: _Scope | None = scope
        while current is not None:
            if ref.table is not None:
                for source in current.sources:
                    if source.key == ref.table.key:
                        for column_key, name in source.columns:
                            if column_key == key:
                                return self.column_text(source, name)
                        return None
            else:
                matches = [
                    (source, name)
                    for source in current.sources
                    for column_key, name in source.columns
                    if column_key == key
                ]
                if len(matches) > 1:
                    raise SQLValidationError(
                        f"column '{ref.column.name}' is ambiguous; qualify it with a table "
                        f"name or alias",
                        code="ambiguous",
                    )
                if matches:
                    return self.column_text(*matches[0])
            current = current.parent
        return None

    # --- expressions ---

    def expr(
        self, expr: Expr, ctx: _Scope | _FilterScope, *, use_aliases: bool = False
    ) -> str:
        e = lambda sub: self.expr(sub, ctx, use_aliases=use_aliases)

        if isinstance(expr, Literal):
            if expr.kind == "string":
                return self.param(expr.value)
            if expr.kind == "number":
                # Numbers are inlined: the tokenizer admits only digits, a point and an
                # exponent, so there is nothing to inject, and an inlined number keeps
                # "GROUP BY x + 1" matching "SELECT x + 1" in databases that compare
                # the two textually (two bind parameters never match).
                # The token text, not repr(value): repr of an overflowing float is "inf",
                # which a database would read as a column name.
                return expr.text
            if expr.kind == "bool":
                return "TRUE" if expr.value else "FALSE"
            return "NULL"
        if isinstance(expr, ColumnRef):
            return self.column_ref(expr, ctx, use_aliases)
        if isinstance(expr, Param):
            if not isinstance(ctx, _FilterScope):
                raise SQLValidationError("bind parameters are not allowed", code="syntax")
            if expr.name not in ctx.row_filter.params:
                raise ConfigError(
                    f"The row filter for '{ctx.entry.name}' uses :{expr.name} but its "
                    f"params do not define it."
                )
            return self.param(ctx.row_filter.params[expr.name])
        if isinstance(expr, Unary):
            if expr.op == "NOT":
                return f"(NOT {e(expr.operand)})"
            return f"({expr.op}{e(expr.operand)})"
        if isinstance(expr, Binary):
            op = "%%" if expr.op == "%" and self.dialect.percent_escaped else expr.op
            return f"({e(expr.left)} {op} {e(expr.right)})"
        if isinstance(expr, Is):
            return f"({e(expr.left)} IS {'NOT ' if expr.negated else ''}{e(expr.right)})"
        if isinstance(expr, Like):
            text = f"({e(expr.operand)} {'NOT ' if expr.negated else ''}LIKE {e(expr.pattern)}"
            if expr.escape is not None:
                text += f" ESCAPE {e(expr.escape)}"
            return text + ")"
        if isinstance(expr, Between):
            return (
                f"({e(expr.operand)} {'NOT ' if expr.negated else ''}BETWEEN "
                f"{e(expr.low)} AND {e(expr.high)})"
            )
        if isinstance(expr, InList):
            items = ", ".join(e(item) for item in expr.items)
            return f"({e(expr.operand)} {'NOT ' if expr.negated else ''}IN ({items}))"
        if isinstance(expr, (InQuery, Exists, ScalarSubquery)):
            if isinstance(ctx, _FilterScope):
                raise ConfigError(
                    f"The row filter for '{ctx.entry.name}' uses a subquery; row filters "
                    f"may only use the table's own columns."
                )
            body, names, _, _ = self.query(expr.query, ctx)
            if isinstance(expr, Exists):
                return f"(EXISTS ({body}))"
            if len(names) != 1:
                raise SQLValidationError(
                    "a subquery used as a value or with IN must return exactly one column",
                    code="column_count",
                )
            if isinstance(expr, InQuery):
                return f"({e(expr.operand)} {'NOT ' if expr.negated else ''}IN ({body}))"
            return f"({body})"
        if isinstance(expr, FuncCall):
            return self.call(expr, e)
        if isinstance(expr, Case):
            text = "(CASE"
            if expr.operand is not None:
                text += f" {e(expr.operand)}"
            for condition, result in expr.whens:
                text += f" WHEN {e(condition)} THEN {e(result)}"
            if expr.default is not None:
                text += f" ELSE {e(expr.default)}"
            return text + " END)"
        if isinstance(expr, Cast):
            base = expr.type_name.split("(", 1)[0].strip()
            if base not in self.dialect.cast_types:
                raise SQLValidationError(
                    f"CAST to {base} is not allowed", code="type_not_allowed"
                )
            return f"CAST({e(expr.operand)} AS {expr.type_name})"
        if isinstance(expr, Star):
            raise SQLValidationError(
                "'*' is only allowed as a whole result column or in COUNT(*)", code="syntax"
            )
        raise SQLValidationError("unsupported expression", code="unsupported")

    def call(self, call: FuncCall, e: Any) -> str:
        allowed = self.dialect.niladic if call.niladic else self.dialect.functions
        if call.name not in allowed:
            raise SQLValidationError(
                f"function {call.name} is not allowed", code="function_not_allowed"
            )
        if call.niladic:
            return call.name
        if call.star:
            if call.name != "COUNT":
                raise SQLValidationError(f"{call.name}(*) is not valid", code="syntax")
            return "COUNT(*)"
        amplifier = call.name in _AMPLIFIERS
        if amplifier and self.amplifier_depth:
            raise SQLValidationError(
                f"{call.name} may not be nested inside another {call.name}",
                code="too_complex",
            )
        if self.func_depth >= MAX_FUNCTION_DEPTH:
            raise SQLValidationError(
                f"function calls nest more than {MAX_FUNCTION_DEPTH} deep",
                code="too_complex",
            )
        self.func_depth += 1
        self.amplifier_depth += amplifier
        try:
            args = ", ".join(e(arg) for arg in call.args)
        finally:
            self.func_depth -= 1
            self.amplifier_depth -= amplifier
        return f"{call.name}({'DISTINCT ' if call.distinct else ''}{args})"

    def column_ref(self, ref: ColumnRef, ctx: _Scope | _FilterScope, use_aliases: bool) -> str:
        if isinstance(ctx, _FilterScope):
            if ref.table is not None and ref.table.key != ctx.entry.name.casefold():
                raise ConfigError(
                    f"The row filter for '{ctx.entry.name}' refers to another table "
                    f"('{ref.table.name}')."
                )
            for name in ctx.entry.base_columns:
                if name.casefold() == ref.column.key:
                    ctx.used.add(name)
                    return self.q(name)
            raise ConfigError(
                f"The row filter for '{ctx.entry.name}' uses '{ref.column.name}', which is "
                f"not a column of that table."
            )

        resolved = self.resolve(ref, ctx)
        if resolved is not None:
            return resolved
        if use_aliases and ref.table is None and ref.column.key in ctx.aliases:
            # A result-column alias used in WHERE/GROUP BY/HAVING: substitute the
            # aliased expression rather than relying on each database's alias rules.
            return f"({self.expr(ctx.aliases[ref.column.key], ctx)})"
        written = f"{ref.table.name}.{ref.column.name}" if ref.table else ref.column.name
        raise SQLValidationError(
            f"unknown column '{written}'", code="unknown_column", reference=written
        )

    def filter_predicate(self, entry: EntitledTable, row_filter: RowFilter) -> str:
        try:
            predicate = parse_predicate(row_filter.predicate)
        except SQLSyntaxError as exc:
            raise ConfigError(
                f"The row filter for '{entry.name}' is not a valid predicate: {exc}"
            ) from None
        ctx = _FilterScope(entry, row_filter)
        text = self.expr(predicate, ctx)
        self.read_set.setdefault(entry.name, set()).update(ctx.used)
        return text
