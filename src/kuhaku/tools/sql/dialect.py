"""What differs between databases in the SQL the validator renders.

Three things do: which functions may be called, which type names ``CAST`` may name,
and how bind parameters are written. Identifier quoting (``"name"``), ``LIMIT``/``OFFSET`` and every operator in
the accepted subset are the same in SQLite and PostgreSQL.
"""

from __future__ import annotations

from dataclasses import dataclass

from kuhaku.core.exceptions import ConfigError

PARAMSTYLES = frozenset({"qmark", "format", "numeric", "named", "pyformat"})

_COMMON_FUNCTIONS = frozenset(
    {
        "COUNT", "SUM", "AVG", "MIN", "MAX", "ABS", "ROUND", "LOWER", "UPPER", "LENGTH",
        "SUBSTR", "TRIM", "LTRIM", "RTRIM", "COALESCE", "NULLIF", "REPLACE",
    }
)

# Type names CAST may target (the base name; a size such as VARCHAR(20) is allowed on
# top). An allow list for the same reason as functions: PostgreSQL's reg* types
# (CAST('pg_shadow' AS regclass)) turn a cast into a catalog lookup.
_COMMON_CAST_TYPES = frozenset(
    {
        "INTEGER", "INT", "BIGINT", "SMALLINT", "REAL", "FLOAT", "DOUBLE PRECISION",
        "NUMERIC", "DECIMAL", "TEXT", "VARCHAR", "CHAR", "CHARACTER", "CHARACTER VARYING",
        "DATE", "TIMESTAMP", "BOOLEAN",
    }
)


@dataclass(frozen=True)
class Dialect:
    """``functions`` is an allow list: a function not in it is rejected, whatever it
    does. That is what keeps ``load_extension``, ``pg_read_file``, ``dblink`` and their
    like out without having to know about each of them. ``cast_types`` is the same for
    the type names ``CAST`` may name (upper case, base name without size)."""

    name: str
    functions: frozenset[str]
    niladic: frozenset[str] = frozenset({"CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP"})
    paramstyle: str = "qmark"
    cast_types: frozenset[str] = _COMMON_CAST_TYPES

    def __post_init__(self) -> None:
        if self.paramstyle not in PARAMSTYLES:
            raise ConfigError(
                f"Unknown paramstyle '{self.paramstyle}'; expected one of {sorted(PARAMSTYLES)}."
            )

    @property
    def percent_escaped(self) -> bool:
        # With format/pyformat paramstyles the driver treats every "%" in the text as
        # the start of a placeholder, so a literal one (the modulo operator, a quoted
        # name) has to be doubled.
        return self.paramstyle in ("format", "pyformat")


SQLITE = Dialect(
    "sqlite",
    _COMMON_FUNCTIONS
    | frozenset(
        {
            "TOTAL", "GROUP_CONCAT", "IFNULL", "IIF", "INSTR", "DATE", "TIME", "DATETIME",
            "JULIANDAY", "STRFTIME", "TYPEOF",
        }
    ),
)

POSTGRES = Dialect(
    "postgresql",
    _COMMON_FUNCTIONS
    | frozenset({"SUBSTRING", "STRING_AGG", "DATE_TRUNC", "TO_CHAR", "GREATEST", "LEAST"}),
    paramstyle="format",
    cast_types=_COMMON_CAST_TYPES | frozenset({"TIME", "TIMESTAMPTZ", "INTERVAL"}),
)
