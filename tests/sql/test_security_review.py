"""Regression tests for the text-to-SQL security review findings."""

from __future__ import annotations

import sqlite3
import time

import pytest

from kuhaku.tools.sql import (
    POSTGRES,
    SQLITE,
    AccessPolicy,
    Column,
    DBAPIDatabase,
    QueryExecutionError,
    SchemaCatalog,
    SQLiteDatabase,
    SQLValidationError,
    Table,
    Validator,
    entitled_schema,
)

from .conftest import ENG, tenant_filter

CATALOG = SchemaCatalog(
    [
        Table("depts", (Column("id"), Column("title"))),
        Table("orders", (Column("id"), Column("tenant"), Column("customer"), Column("secret"))),
    ]
)
POLICY = AccessPolicy(columns={"orders.secret": ["hr"]}, row_filters={"orders": tenant_filter})


def validator(dialect=SQLITE) -> Validator:
    return Validator(
        entitled_schema(CATALOG, POLICY, ENG), dialect=dialect, reserved_names=CATALOG.names
    )


def rejected(sql: str, dialect=SQLITE) -> SQLValidationError:
    with pytest.raises(SQLValidationError) as info:
        validator(dialect).validate(sql)
    return info.value


# --- finding 4: the caller-facing SQL hides the row filter ----------------------------


def test_display_sql_abstracts_the_row_filter_and_drops_its_params():
    result = validator().validate("SELECT customer FROM orders WHERE customer = 'x'")
    assert "tenant\" = ?" in result.sql and result.params == ("acme", "x")
    assert "/* row filter */ TRUE" in result.display_sql
    assert "acme" not in str(result.display_params) and result.display_params == ("x",)
    # Same shape otherwise: the executed SQL differs only in the filter.
    assert result.display_sql.replace("/* row filter */ TRUE", "") != result.display_sql


def test_display_sql_with_a_filter_on_a_hidden_column():
    policy = AccessPolicy(
        columns={"orders.tenant": ["hr"], "orders.secret": ["hr"]},
        row_filters={"orders": tenant_filter},
    )
    v = Validator(entitled_schema(CATALOG, policy, ENG), reserved_names=CATALOG.names)
    result = v.validate("SELECT customer FROM orders")
    assert '"tenant"' in result.sql  # the filter reads the hidden column...
    assert "tenant" not in result.display_sql  # ...the caller is never told
    assert result.display_params == ()


# --- finding 5: amplification -----------------------------------------------------------


def test_nested_replace_is_rejected():
    # The reviewer's repro shape: each REPLACE level multiplies the string.
    sql = "SELECT " + "REPLACE(" * 3 + "'a'" + ", 'a', 'aaaaaaaaaa')" * 3
    assert rejected(sql).code == "too_complex"
    assert validator().validate("SELECT REPLACE(title, 'a', 'b') FROM depts")


def test_function_nesting_depth_is_bounded():
    deep = "SELECT " + "LOWER(" * 7 + "title" + ")" * 7 + " FROM depts"
    assert rejected(deep).code == "too_complex"
    ok = "SELECT " + "LOWER(" * 6 + "title" + ")" * 6 + " FROM depts"
    assert validator().validate(ok)


def test_sqlite_value_length_is_capped_per_query():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE depts (id INTEGER, title TEXT)")
    connection.execute("INSERT INTO depts VALUES (1, 'aaaaaaaaaa')")
    database = SQLiteDatabase.from_connection(connection, max_value_bytes=1_000)
    catalog = SchemaCatalog.from_database(database)
    v = Validator(entitled_schema(catalog, AccessPolicy(), None))
    big = v.validate("SELECT REPLACE(title, 'a', '" + "b" * 200 + "') FROM depts")
    start = time.monotonic()
    with pytest.raises(QueryExecutionError):
        database.execute(big, max_rows=5, timeout_seconds=0.5)
    assert time.monotonic() - start < 5
    small = v.validate("SELECT REPLACE(title, 'a', 'bb') FROM depts")
    assert database.execute(small, max_rows=5, timeout_seconds=1).rows == (("b" * 20,),)
    # The connection's own limit is restored afterwards.
    assert connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH) > 1_000


def test_total_result_bytes_are_capped():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE depts (id INTEGER, title TEXT)")
    connection.executemany("INSERT INTO depts VALUES (?, ?)", [(i, "x" * 100) for i in range(50)])
    database = SQLiteDatabase.from_connection(connection, max_result_bytes=1_000)
    catalog = SchemaCatalog.from_database(database)
    query = Validator(entitled_schema(catalog, AccessPolicy(), None)).validate(
        "SELECT title FROM depts"
    )
    rows = database.execute(query, max_rows=50, timeout_seconds=1)
    assert rows.truncated and 0 < len(rows.rows) < 50


# --- finding 6: CAST type allow-list ----------------------------------------------------


@pytest.mark.parametrize("type_name", ["regclass", "regrole", "regproc", "oid", "pg_lsn"])
def test_postgres_catalog_oracle_types_are_rejected(type_name):
    sql = f"SELECT CAST('pg_shadow' AS {type_name}) FROM depts"
    assert rejected(sql, POSTGRES).code == "type_not_allowed"


def test_ordinary_casts_still_work():
    for sql in (
        "SELECT CAST(id AS TEXT) FROM depts",
        "SELECT CAST(title AS VARCHAR(20)) FROM depts",
        "SELECT CAST(id AS DOUBLE PRECISION) FROM depts",
    ):
        assert validator(POSTGRES).validate(sql)
    assert rejected("SELECT CAST(id AS INTERVAL) FROM depts").code == "type_not_allowed"


# --- finding 11: control characters in quoted identifiers -------------------------------


@pytest.mark.parametrize("char", ["\x00", "\x01", "\n", "\x7f"])
def test_control_characters_in_quoted_identifiers_are_rejected(char):
    assert rejected(f'SELECT "ti{char}tle" FROM depts').code == "syntax"
    assert rejected('SELECT "a\x000\x00" FROM depts').code == "syntax"


# --- finding 12: views and huge LIMIT/OFFSET --------------------------------------------


def test_views_are_not_introspected(tmp_path):
    path = tmp_path / "v.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE t (a INTEGER); CREATE VIEW v AS SELECT a FROM t;"
    )
    connection.close()
    assert [t.name for t in SQLiteDatabase(path).introspect()] == ["t"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT title FROM depts LIMIT 9223372036854775808",
        "SELECT title FROM depts LIMIT 1 OFFSET 99999999999999999999999",
        "SELECT title FROM depts LIMIT 1 OFFSET " + "9" * 5000,
    ],
)
def test_huge_limit_or_offset_is_rejected(sql):
    assert rejected(sql).code == "limit"


def test_huge_number_literal_is_a_clean_rejection():
    assert rejected("SELECT " + "9" * 5000 + " FROM depts").code == "syntax"


def test_largest_offset_is_accepted():
    result = validator().validate("SELECT title FROM depts LIMIT 1 OFFSET 9223372036854775807")
    assert result.sql.endswith("OFFSET 9223372036854775807")


# --- finding 13: DBAPIDatabase warns without setup --------------------------------------


def test_dbapi_without_setup_warns():
    with pytest.warns(UserWarning, match="statement timeout"):
        DBAPIDatabase(lambda: None, dialect=SQLITE, tables=[Table("t", (Column("a"),))])
