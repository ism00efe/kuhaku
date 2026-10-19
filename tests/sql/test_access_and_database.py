"""Access-policy semantics, catalog building, and the database adapters' second line of
defence (read-only connection, authorizer, timeout, row cap)."""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace

import pytest

from kuhaku.core.auth import AuthContext
from kuhaku.core.exceptions import ConfigError
from kuhaku.tools.sql import (
    AccessPolicy,
    Column,
    DBAPIDatabase,
    PostgresDatabase,
    QueryExecutionError,
    RowFilter,
    SchemaCatalog,
    SQLiteDatabase,
    Table,
    Validator,
    entitled_schema,
    is_entitled,
)

from .conftest import ENG, HR

CATALOG = SchemaCatalog(
    [Table("t", (Column("a"), Column("b"))), Table("u", (Column("c"),))]
)


# --- the entitlement rule ----------------------------------------------------------


@pytest.mark.parametrize(
    "tags, auth, visible",
    [
        ((), None, True),
        ((), ENG, True),
        (("hr",), None, False),
        (("hr",), ENG, False),
        (("hr",), HR, True),
        (("HR",), HR, False),  # exact, case-sensitive strings, as in RAG
        (("hr", "eng"), ENG, True),
        (("hr",), AuthContext.anonymous(), False),
    ],
)
def test_entitlement_rule_matches_rag(tags, auth, visible):
    assert is_entitled(tags, auth) is visible


def test_policy_names_are_checked_against_the_catalog():
    with pytest.raises(ConfigError, match="'nosuch'"):
        AccessPolicy(tables={"nosuch": ["x"]}).check(CATALOG)
    with pytest.raises(ConfigError, match="'t.z'"):
        AccessPolicy(columns={"t.z": ["x"]}).check(CATALOG)
    with pytest.raises(ConfigError):
        AccessPolicy(row_filters={"nosuch": RowFilter("1 = 1")}).check(CATALOG)
    with pytest.raises(ConfigError):
        AccessPolicy(columns={"no_dot": ["x"]})


@pytest.mark.parametrize("tags", [[""], ["  "], ["ok", ""]])
def test_empty_tags_are_rejected(tags):
    with pytest.raises(ValueError):
        AccessPolicy(tables={"t": tags})


def test_a_tag_string_instead_of_a_list_is_rejected():
    with pytest.raises(ConfigError):
        AccessPolicy(tables={"t": "hr"})


def test_policy_keys_match_table_names_without_regard_to_case():
    schema = entitled_schema(CATALOG, AccessPolicy(tables={"T": ["hr"]}), ENG)
    assert schema.names == ("u",)


def test_a_table_with_no_visible_column_is_invisible():
    policy = AccessPolicy(columns={"u.c": ["hr"]})
    assert entitled_schema(CATALOG, policy, ENG).names == ("t",)
    assert entitled_schema(CATALOG, policy, HR).names == ("t", "u")


def test_a_row_filter_source_returning_the_wrong_type_fails():
    policy = AccessPolicy(row_filters={"t": lambda auth: "a = 1"})
    with pytest.raises(ConfigError):
        entitled_schema(CATALOG, policy, ENG)


def test_sample_values_are_withheld_for_row_filtered_tables():
    catalog = CATALOG.describe(sample_values={"t.a": ["x", "y"], "u.c": ["z"]})
    schema = entitled_schema(catalog, AccessPolicy(row_filters={"t": RowFilter("1 = 1")}), None)
    assert schema.get("t").table.column("a").sample_values == ()
    assert schema.get("u").table.column("c").sample_values == ("z",)


def test_describe_rejects_typos():
    with pytest.raises(ConfigError):
        CATALOG.describe({"nosuch": "x"})
    with pytest.raises(ConfigError):
        CATALOG.describe(columns={"t.nosuch": "x"})
    described = CATALOG.describe({"t": "a table"}, columns={"t.a": "a column"})
    assert described.get("t").description == "a table"
    assert described.get("t").column("a").description == "a column"


def test_catalog_rejects_duplicate_names():
    with pytest.raises(ConfigError):
        SchemaCatalog([Table("t", (Column("a"),)), Table("T", (Column("a"),))])
    with pytest.raises(ConfigError):
        Table("t", (Column("a"), Column("A")))


# --- SQLite adapter ----------------------------------------------------------------


def test_introspection_and_sample_values(database):
    catalog = SchemaCatalog.from_database(database, sample_values=2)
    assert set(catalog.names) == {"depts", "employees", "salaries", "orders"}
    employees = catalog.get("employees")
    assert [c.name for c in employees.columns][:3] == ["id", "name", "dept_id"]
    assert employees.column("salary").type == "INTEGER"
    assert len(employees.column("name").sample_values) == 2
    assert employees.column("salary").sample_values == ()  # not a text column


def test_introspection_can_be_limited_to_named_tables(database):
    catalog = SchemaCatalog.from_database(database, tables=["depts"])
    assert catalog.names == ("depts",)
    with pytest.raises(ConfigError):
        SchemaCatalog.from_database(database, tables=["nosuch"])


def _validated(database, sql, auth=ENG, policy=None):
    catalog = SchemaCatalog.from_database(database)
    schema = entitled_schema(catalog, policy or AccessPolicy(), auth)
    return Validator(schema, max_rows=100, reserved_names=catalog.names).validate(sql)


def test_executes_read_only(database):
    rows = database.execute(_validated(database, "SELECT title FROM depts ORDER BY id"),
                            max_rows=10, timeout_seconds=5)
    assert rows.columns == ("title",)
    assert rows.rows == (("Engineering",), ("People",))
    assert not rows.truncated

    write = replace(_validated(database, "SELECT 1"), sql="DELETE FROM depts", params=())
    with pytest.raises(QueryExecutionError):
        database.execute(write, max_rows=10, timeout_seconds=5)
    check = database.execute(_validated(database, "SELECT count(*) AS n FROM depts"),
                             max_rows=10, timeout_seconds=5)
    assert check.rows == ((2,),)


def test_the_authorizer_denies_reads_outside_the_read_set(database):
    # A query the validator never produced, as if it had a bug: it reads a column the
    # read set does not contain, and SQLite itself refuses.
    hide = AccessPolicy(tables={"salaries": ["hr"]}, columns={"employees.salary": ["hr"]})
    validated = _validated(database, "SELECT name FROM employees", policy=hide)
    smuggled = replace(validated, sql='SELECT "salary" FROM "employees"')
    with pytest.raises(QueryExecutionError, match="not authorized|prohibited"):
        database.execute(smuggled, max_rows=10, timeout_seconds=5)
    other_table = replace(validated, sql='SELECT "amount" FROM "salaries"')
    with pytest.raises(QueryExecutionError, match="not authorized|prohibited"):
        database.execute(other_table, max_rows=10, timeout_seconds=5)
    for sql in ("SELECT name FROM sqlite_master", "SELECT randomblob(4)",
                "SELECT load_extension('x')"):
        with pytest.raises(QueryExecutionError):
            database.execute(replace(validated, sql=sql), max_rows=10, timeout_seconds=5)


def test_row_cap_and_truncation(database):
    rows = database.execute(_validated(database, "SELECT name FROM employees"),
                            max_rows=2, timeout_seconds=5)
    assert len(rows.rows) == 2 and rows.truncated


def test_a_long_query_is_interrupted(database):
    slow = _validated(
        database,
        "SELECT count(*) FROM employees a, employees b, employees c, employees d, "
        "employees e, employees f, employees g, employees h, employees i, employees j, "
        "employees k, employees l, employees m, employees n",
    )
    with pytest.raises(QueryExecutionError) as info:
        database.execute(slow, max_rows=10, timeout_seconds=0.05)
    assert info.value.timed_out


def test_borrowed_connection_keeps_authorizer_and_is_restored():
    connection = sqlite3.connect(":memory:")
    connection.executescript("CREATE TABLE t (a, b); INSERT INTO t VALUES (1, 2);")
    database = SQLiteDatabase.from_connection(connection)
    validated = _validated(database, "SELECT a FROM t",
                           policy=AccessPolicy(columns={"t.b": ["hr"]}))
    assert database.execute(validated, max_rows=5, timeout_seconds=5).rows == ((1,),)
    with pytest.raises(QueryExecutionError):
        database.execute(replace(validated, sql="SELECT b FROM t"), max_rows=5,
                         timeout_seconds=5)
    with pytest.raises(QueryExecutionError):
        database.execute(replace(validated, sql="DROP TABLE t"), max_rows=5,
                         timeout_seconds=5)
    # The caller's connection is unrestricted again afterwards.
    assert connection.execute("SELECT b FROM t").fetchall() == [(2,)]


def test_sqlite_adapter_refuses_memory_and_missing_files(tmp_path):
    with pytest.raises(ConfigError):
        SQLiteDatabase(":memory:")
    with pytest.raises(ConfigError):
        SQLiteDatabase(tmp_path / "missing.db")


# --- DB-API and PostgreSQL adapters ------------------------------------------------


def test_dbapi_adapter_runs_and_rolls_back(db_path):
    from kuhaku.tools.sql import SQLITE

    setups = []
    database = DBAPIDatabase(
        lambda: sqlite3.connect(db_path),
        dialect=SQLITE,
        tables=[Table("depts", (Column("id"), Column("title")))],
        setup=lambda cursor, timeout: setups.append(timeout),
    )
    catalog = SchemaCatalog.from_database(database)
    validated = Validator(entitled_schema(catalog, AccessPolicy(), None)).validate(
        "SELECT title FROM depts WHERE id = 1"
    )
    rows = database.execute(validated, max_rows=5, timeout_seconds=3)
    assert rows.rows == (("Engineering",),) and setups == [3]
    with pytest.raises(ConfigError):
        DBAPIDatabase(lambda: None, dialect=SQLITE).introspect()


def test_postgres_adapter_imports_its_driver_lazily(monkeypatch):
    import kuhaku.tools.sql  # noqa: F401 -- importing the package must not need psycopg

    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(ConfigError, match="sql-postgres"):
        PostgresDatabase("postgresql://localhost/x")


def test_postgres_adapter_sets_read_only_and_timeout(monkeypatch):
    executed: list[tuple[str, object]] = []

    class Cursor:
        description = (("n",),)

        def execute(self, sql, params=None):
            executed.append((sql, params))

        def fetchmany(self, size):
            return [(1,)]

        def fetchall(self):
            return [("t", "a", "integer"), ("t", "b", "text")]

    class Connection:
        def cursor(self):
            return Cursor()

        def rollback(self):
            executed.append(("ROLLBACK", None))

        def close(self):
            pass

    class FakePsycopg:
        @staticmethod
        def connect(conninfo):
            return Connection()

    monkeypatch.setitem(sys.modules, "psycopg", FakePsycopg)
    database = PostgresDatabase("postgresql://x")
    catalog = SchemaCatalog.from_database(database)
    assert [c.name for c in catalog.get("t").columns] == ["a", "b"]
    executed.clear()
    validated = Validator(
        entitled_schema(catalog, AccessPolicy(), None), dialect=database.dialect
    ).validate("SELECT a FROM t WHERE b = 'x'")
    rows = database.execute(validated, max_rows=5, timeout_seconds=2)
    assert rows.rows == ((1,),)
    assert executed[0][0] == "SET TRANSACTION READ ONLY"
    assert executed[1][0] == "SET LOCAL statement_timeout = 2000"
    assert executed[2][0] == 'SET LOCAL search_path TO "public", pg_catalog, pg_temp'
    assert executed[3] == (validated.sql, ("x",)) and "%s" in validated.sql
    assert executed[-1][0] == "ROLLBACK"
