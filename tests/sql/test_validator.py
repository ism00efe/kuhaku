"""The validator is the security core: what it accepts, what it rejects, and that a
non-entitled name is rejected exactly like a nonexistent one."""

from __future__ import annotations

import sqlite3

import pytest

from kuhaku.core.exceptions import ConfigError
from kuhaku.tools.sql import (
    POSTGRES,
    SQLITE,
    AccessPolicy,
    Column,
    Dialect,
    RowFilter,
    SchemaCatalog,
    SQLValidationError,
    Table,
    Validator,
    entitled_schema,
)

from .conftest import ENG, HR, tenant_filter

CATALOG = SchemaCatalog(
    [
        Table("depts", (Column("id"), Column("title"))),
        Table(
            "employees",
            (
                Column("id"),
                Column("name"),
                Column("dept_id"),
                Column("email"),
                Column("salary"),
                Column("tenant"),
            ),
        ),
        Table("salaries", (Column("emp_id"), Column("amount"))),
        Table("orders", (Column("id"), Column("tenant"), Column("customer"), Column("total"))),
    ]
)
POLICY = AccessPolicy(
    tables={"salaries": ["hr"]},
    columns={"employees.salary": ["hr"], "employees.email": ["hr"]},
    row_filters={"orders": tenant_filter},
)


def validator(auth=ENG, *, policy=POLICY, max_rows=100, dialect=SQLITE) -> Validator:
    return Validator(
        entitled_schema(CATALOG, policy, auth),
        dialect=dialect,
        max_rows=max_rows,
        reserved_names=CATALOG.names,
    )


def rejected(sql: str, auth=ENG, **kwargs) -> SQLValidationError:
    with pytest.raises(SQLValidationError) as info:
        validator(auth, **kwargs).validate(sql)
    return info.value


# --- statement shape ---------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, code",
    [
        ("SELECT 1; DROP TABLE depts", "multiple_statements"),
        ("SELECT 1;; ", "multiple_statements"),
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("DELETE FROM depts", "not_select"),
        ("INSERT INTO depts VALUES (3, 'x')", "not_select"),
        ("UPDATE depts SET title = 'x'", "not_select"),
        ("DROP TABLE depts", "not_select"),
        ("CREATE TABLE x (a)", "not_select"),
        ("PRAGMA table_info(depts)", "not_select"),
        ("ATTACH DATABASE 'x.db' AS x", "not_select"),
        ("VACUUM", "not_select"),
        ("REPLACE INTO depts VALUES (1, 'x')", "not_select"),
        ("EXPLAIN SELECT 1", "not_select"),
        ("VALUES (1)", "not_select"),
        ("SELECT title FROM depts -- AND tenant = 'acme'", "comment"),
        ("SELECT title /* hi */ FROM depts", "comment"),
        ("SELECT title FROM depts WHERE id = ?", "syntax"),
        ("SELECT title FROM depts WHERE id = :id", "syntax"),
        ("SELECT title FROM depts WHERE id = @id", "syntax"),
        ("SELECT title FROM depts WHERE id = $1", "syntax"),
        ("SELECT x'41'", "syntax"),
        ("SELECT 0x41", "syntax"),
        ("SELECT title INTO newtable FROM depts", "not_select"),
        ("WITH RECURSIVE r AS (SELECT 1) SELECT * FROM r", "unsupported"),
        ("SELECT count(*) OVER () FROM depts", "unsupported"),
        ("SELECT * FROM depts NATURAL JOIN orders", "unsupported"),
        ("SELECT * FROM depts JOIN orders USING (id)", "unsupported"),
        ("SELECT * FROM depts RIGHT JOIN orders ON 1 = 1", "unsupported"),
        ("SELECT title FROM depts LIMIT -1", "limit"),
        ("SELECT title FROM depts LIMIT 1 + 1", "limit"),
        ("SELECT title FROM depts LIMIT 5, 10", "unsupported"),
        ("SELECT title FROM main.depts", "unknown_table"),
        ("SELECT title FROM depts WHERE title GLOB 'E*'", "unsupported"),
        ("SELECT title FROM depts ORDER BY title COLLATE NOCASE", "unsupported"),
        ("SELECT load_extension('evil.so')", "function_not_allowed"),
        ("SELECT sqlite_version()", "function_not_allowed"),
        ("SELECT title FROM depts WHERE randomblob(10) IS NOT NULL", "function_not_allowed"),
        ("", "syntax"),
        ("SELECT title FROM depts\x00", "syntax"),
    ],
)
def test_rejects_everything_outside_the_read_only_subset(sql, code):
    assert rejected(sql).code == code


def test_trailing_semicolon_and_semicolon_in_a_string_are_fine():
    assert validator().validate("SELECT title FROM depts;").params == ()
    result = validator().validate("SELECT title FROM depts WHERE title = 'a; DROP TABLE x'")
    assert result.params == ("a; DROP TABLE x",)
    assert "DROP" not in result.sql


def test_length_and_depth_limits_are_clean_rejections():
    assert rejected("SELECT " + "1 + " * 6000 + "1").code == "too_long"
    assert rejected("SELECT " + "(" * 200 + "1" + ")" * 200).code == "too_deep"


# --- what is rendered --------------------------------------------------------------


def test_the_executed_sql_is_rendered_not_passed_through():
    result = validator().validate(
        "select e.name, d.title from employees e join depts d on e.dept_id = d.id "
        "where e.name like 'A%' order by name"
    )
    assert result.sql == (
        'SELECT "t1"."name" AS "name", "t2"."title" AS "title" '
        'FROM "employees" AS "t1" JOIN "depts" AS "t2" ON ("t1"."dept_id" = "t2"."id") '
        'WHERE ("t1"."name" LIKE ?) ORDER BY 1 LIMIT 100'
    )
    assert result.params == ("A%",)
    assert result.tables == ("employees", "depts")
    assert set(result.columns) == {
        "employees.name", "employees.dept_id", "depts.title", "depts.id"
    }


def test_string_literals_are_parameters_so_quotes_cannot_break_out():
    result = validator().validate("SELECT title FROM depts WHERE title = 'x'' OR ''1''=''1'")
    assert result.params == ("x' OR '1'='1",)
    assert "OR" not in result.sql


def test_numbers_are_inlined_from_their_token_text():
    result = validator().validate("SELECT 1e999, 2.5, 3")
    assert result.sql.startswith('SELECT 1e999 AS "1e999", 2.5 AS "2.5", 3 AS "3"')
    assert result.params == ()


def test_star_expands_to_entitled_columns_only():
    result = validator().validate("SELECT * FROM employees")
    assert result.output_columns == ("id", "name", "dept_id", "tenant")
    assert "salary" not in result.sql and "email" not in result.sql
    assert validator(HR).validate("SELECT * FROM employees").output_columns == (
        "id", "name", "dept_id", "email", "salary", "tenant",
    )


def test_qualified_star_and_star_without_from():
    result = validator().validate("SELECT d.* FROM depts d JOIN employees e ON e.dept_id = d.id")
    assert result.output_columns == ("id", "title")
    assert rejected("SELECT *").code == "unsupported"
    assert rejected("SELECT x.* FROM depts").code == "unknown_table"


def test_limit_is_capped_and_a_smaller_limit_kept():
    capped = validator(max_rows=10).validate("SELECT title FROM depts LIMIT 1000")
    assert capped.sql.endswith("LIMIT 10") and capped.limit_capped
    kept = validator(max_rows=10).validate("SELECT title FROM depts LIMIT 3 OFFSET 1")
    assert kept.sql.endswith("LIMIT 3 OFFSET 1") and not kept.limit_capped
    missing = validator(max_rows=10).validate("SELECT title FROM depts")
    assert missing.sql.endswith("LIMIT 10") and missing.limit_capped


def test_limit_applies_to_the_whole_compound_query():
    result = validator(max_rows=5).validate(
        "SELECT title FROM depts UNION ALL SELECT customer FROM orders"
    )
    assert result.sql.endswith("LIMIT 5")


def test_operators_are_fully_parenthesized():
    result = validator().validate("SELECT id FROM depts WHERE id = 1 OR id = 2 AND title = 'x'")
    assert '(("t1"."id" = 1) OR (("t1"."id" = 2) AND ("t1"."title" = ?)))' in result.sql


def test_format_paramstyle_escapes_percent():
    result = validator(dialect=POSTGRES).validate(
        "SELECT id % 2 AS \"50%\" FROM depts WHERE title LIKE 'a%'"
    )
    assert '("t1"."id" %% 2) AS "50%%"' in result.sql
    assert "LIKE %s" in result.sql and result.params == ("a%",)


@pytest.mark.parametrize(
    "style, placeholder, params",
    [
        ("qmark", "?", ("a",)),
        ("numeric", ":1", ("a",)),
        ("named", ":p1", {"p1": "a"}),
        ("pyformat", "%(p1)s", {"p1": "a"}),
    ],
)
def test_paramstyles(style, placeholder, params):
    dialect = Dialect("test", SQLITE.functions, paramstyle=style)
    result = validator(dialect=dialect).validate("SELECT title FROM depts WHERE title = 'a'")
    assert f"= {placeholder})" in result.sql
    assert result.params == params


# --- entitlement -------------------------------------------------------------------

PROBES = [
    # (sql naming something hidden from ENG, same sql naming something nonexistent)
    ("SELECT amount FROM salaries", "SELECT amount FROM nosuch"),
    ("SELECT salary FROM employees", "SELECT bonus FROM employees"),
    ("SELECT e.email FROM employees e", "SELECT e.phone FROM employees e"),
    ('SELECT "SALARY" FROM employees', 'SELECT "BONUS" FROM employees'),
    ("SELECT name FROM employees UNION SELECT emp_id FROM salaries",
     "SELECT name FROM employees UNION SELECT emp_id FROM nosuch"),
    ("SELECT name FROM employees WHERE id IN (SELECT emp_id FROM salaries)",
     "SELECT name FROM employees WHERE id IN (SELECT emp_id FROM nosuch)"),
    ("SELECT name FROM employees WHERE EXISTS (SELECT 1 FROM salaries s WHERE s.emp_id = id)",
     "SELECT name FROM employees WHERE EXISTS (SELECT 1 FROM nosuch s WHERE s.emp_id = id)"),
    ("WITH x AS (SELECT * FROM salaries) SELECT * FROM x",
     "WITH x AS (SELECT * FROM nosuch) SELECT * FROM x"),
    ("SELECT (SELECT max(amount) FROM salaries)", "SELECT (SELECT max(amount) FROM nosuch)"),
    ("SELECT name FROM employees ORDER BY salary", "SELECT name FROM employees ORDER BY bonus"),
    ("SELECT dept_id FROM employees GROUP BY dept_id HAVING max(salary) > 1",
     "SELECT dept_id FROM employees GROUP BY dept_id HAVING max(bonus) > 1"),
    ("SELECT name FROM employees WHERE salary > 100",
     "SELECT name FROM employees WHERE bonus > 100"),
    ("SELECT count(email) FROM employees", "SELECT count(phone) FROM employees"),
    ("SELECT name FROM (SELECT name, salary FROM employees) WHERE salary > 1",
     "SELECT name FROM (SELECT name, bonus FROM employees) WHERE bonus > 1"),
    ('SELECT "Salaries"."amount" FROM "SALARIES"', 'SELECT "Nosuch"."amount" FROM "NOSUCH"'),
    ("SELECT [amount] FROM [salaries]", "SELECT [amount] FROM [nosuch]"),
    ("SELECT `amount` FROM `salaries`", "SELECT `amount` FROM `nosuch`"),
    ("SELECT rowid FROM employees", "SELECT oid2 FROM employees"),
    ("SELECT name FROM sqlite_master", "SELECT name FROM sqlite_nothing"),
]


def _normalize(message: str) -> str:
    for hidden, absent in (
        ("salaries", "nosuch"), ("Salaries", "Nosuch"), ("SALARIES", "NOSUCH"),
        ("salary", "bonus"), ("SALARY", "BONUS"), ("email", "phone"), ("rowid", "oid2"),
        ("sqlite_master", "sqlite_nothing"),
    ):
        message = message.replace(hidden, absent)
    return message


@pytest.mark.parametrize("hidden, absent", PROBES)
def test_a_hidden_name_is_rejected_exactly_like_a_nonexistent_one(hidden, absent):
    hidden_error = rejected(hidden)
    absent_error = rejected(absent)
    assert hidden_error.code == absent_error.code
    assert hidden_error.code in ("unknown_table", "unknown_column")
    # Same message, modulo the name the SQL itself wrote.
    assert _normalize(str(hidden_error)) == str(absent_error)


@pytest.mark.parametrize("hidden, _", PROBES[:14])
def test_the_entitled_caller_is_not_rejected(hidden, _):
    validator(HR).validate(hidden)


def test_an_error_message_names_only_what_the_sql_said():
    error = rejected("SELECT bonus FROM employees")
    for name in ("salary", "email", "salaries", "amount"):
        assert name not in str(error)


def test_no_auth_context_sees_only_untagged_things():
    schema = entitled_schema(CATALOG, POLICY, None)
    assert set(schema.names) == {"depts", "employees", "orders"}
    rejected("SELECT salary FROM employees", auth=None)


def test_case_and_quoting_tricks_resolve_to_the_same_object():
    for sql in (
        "SELECT TITLE FROM DEPTS",
        'SELECT "title" FROM "depts"',
        'SELECT "TiTlE" FROM "DePtS"',
        "SELECT [title] FROM [depts]",
    ):
        result = validator().validate(sql)
        assert result.tables == ("depts",)
        assert result.columns == ("depts.title",)


def test_a_cte_named_like_a_hidden_table_is_only_the_cte():
    result = validator().validate("WITH salaries AS (SELECT 1 AS amount) SELECT * FROM salaries")
    assert result.tables == ()
    assert '"salaries"' not in result.sql


def test_a_cte_that_names_itself_reads_the_table_not_itself():
    # Without WITH RECURSIVE a self-reference must not become recursion: it resolves to
    # the real table -- and here that table is hidden.
    assert rejected("WITH salaries AS (SELECT * FROM salaries) SELECT * FROM salaries").code == (
        "unknown_table"
    )
    result = validator().validate("WITH depts AS (SELECT * FROM depts) SELECT * FROM depts")
    assert result.tables == ("depts",)


def _cte_names(sql: str) -> list[str]:
    import re

    return re.findall(r'(?:WITH|,) "([^"]+)" AS \(', sql)


def test_cte_names_never_collide_with_real_tables(monkeypatch):
    # Force the first random prefix to collide with a (hidden) real table: the query
    # must draw a new prefix rather than reuse or skip past the real name.
    import secrets as secrets_module

    draws = iter(["deadbeef", "0badc0de"])
    monkeypatch.setattr(secrets_module, "token_hex", lambda n: next(draws))
    catalog = SchemaCatalog(
        [Table("qdeadbeef_1", (Column("a"),)), Table("t", (Column("b"),))]
    )
    v = Validator(
        entitled_schema(catalog, AccessPolicy(tables={"qdeadbeef_1": ["x"]}), None),
        reserved_names=catalog.names,
    )
    result = v.validate("WITH c AS (SELECT b FROM t) SELECT * FROM c")
    assert _cte_names(result.sql) == ["q0badc0de_1"]


def test_cte_names_do_not_depend_on_the_catalog():
    # The old scheme numbered cte_1, cte_2 ... skipping real table names, so the gap in
    # the numbering revealed that a hidden table called cte_1 existed.
    sql = "WITH c AS (SELECT b FROM t) SELECT * FROM c"
    shapes = set()
    for hidden in ("cte_1", "cte_2", "other"):
        catalog = SchemaCatalog([Table(hidden, (Column("a"),)), Table("t", (Column("b"),))])
        v = Validator(
            entitled_schema(catalog, AccessPolicy(tables={hidden: ["x"]}), None),
            reserved_names=catalog.names,
        )
        rendered = v.validate(sql).sql
        [name] = _cte_names(rendered)
        assert not name.startswith("cte_")
        shapes.add(rendered.replace(name, "<cte>"))
    assert len(shapes) == 1


def test_correlated_and_scalar_subqueries_resolve_to_the_right_alias():
    result = validator().validate(
        "SELECT name, (SELECT title FROM depts d WHERE d.id = e.dept_id) AS dept "
        "FROM employees e WHERE EXISTS (SELECT 1 FROM orders o WHERE o.tenant = e.tenant)"
    )
    assert '("t2"."id" = "t1"."dept_id")' in result.sql
    assert '"t3"."tenant" = "t1"."tenant"' in result.sql


def test_ambiguous_and_duplicate_names():
    assert rejected("SELECT id FROM depts, employees").code == "ambiguous"
    assert rejected("SELECT 1 FROM depts d, employees d").code == "ambiguous"
    assert rejected("WITH a AS (SELECT 1), a AS (SELECT 2) SELECT 1").code == "ambiguous"


def test_subquery_column_counts():
    assert rejected("SELECT name FROM employees WHERE id IN (SELECT id, title FROM depts)").code \
        == "column_count"
    assert rejected("SELECT id FROM depts UNION SELECT id, name FROM employees").code \
        == "column_count"


def test_order_by_alias_and_ordinal_are_rendered_as_positions():
    result = validator().validate("SELECT title AS t, id FROM depts ORDER BY t DESC, 2")
    assert result.sql.endswith("ORDER BY 1 DESC, 2 LIMIT 100")
    assert rejected("SELECT title FROM depts ORDER BY 3").code == "unsupported"
    assert rejected("SELECT id FROM depts UNION SELECT id FROM orders ORDER BY id + 1").code \
        == "unsupported"


def test_alias_in_where_is_substituted():
    result = validator().validate("SELECT title AS t FROM depts WHERE t = 'x'")
    assert 'WHERE (("t1"."title") = ?)' in result.sql


# --- row filters -------------------------------------------------------------------


def test_row_filter_wraps_every_read_of_the_table():
    result = validator().validate(
        "WITH big AS (SELECT * FROM orders WHERE total > 10) "
        "SELECT customer FROM orders WHERE id IN (SELECT id FROM orders) "
        "UNION SELECT customer FROM big"
    )
    wrapper = '(SELECT "id", "tenant", "customer", "total" FROM "orders" WHERE ("tenant" = ?))'
    assert result.sql.count(wrapper) == 3
    assert result.sql.count('"orders"') == 3  # never read unwrapped
    assert result.params == ("acme", "acme", "acme")


def test_row_filter_cannot_be_escaped_with_or():
    result = validator().validate("SELECT customer FROM orders WHERE 1 = 1 OR tenant <> 'acme'")
    assert 'WHERE ("tenant" = ?)) AS "t1" WHERE' in result.sql


def test_row_filter_may_use_a_hidden_column():
    policy = AccessPolicy(
        columns={"employees.tenant": ["admin"]}, row_filters={"employees": tenant_filter}
    )
    v = validator(policy=policy)
    result = v.validate("SELECT name FROM employees")
    assert '("tenant" = ?)' in result.sql
    assert "tenant" in result.read_set["employees"]
    assert "employees.tenant" not in result.columns
    with pytest.raises(SQLValidationError):
        v.validate("SELECT tenant FROM employees")


@pytest.mark.parametrize(
    "predicate, params",
    [
        ("tenant = :missing", {}),
        ("tenant = 'x'; DROP TABLE orders", {}),
        ("nosuch = 1", {}),
        ("tenant IN (SELECT tenant FROM employees)", {}),
        ("employees.tenant = 1", {}),
    ],
)
def test_bad_row_filters_fail_closed_as_config_errors(predicate, params):
    policy = AccessPolicy(row_filters={"orders": RowFilter(predicate, params)})
    with pytest.raises(ConfigError):
        validator(policy=policy).validate("SELECT customer FROM orders")


def test_the_rendered_sql_runs_and_filters_rows():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE orders (id, tenant, customer, total);"
        "INSERT INTO orders VALUES (1, 'acme', 'a', 1), (2, 'globex', 'b', 2);"
    )
    result = validator().validate("SELECT customer FROM orders")
    assert connection.execute(result.sql, result.params).fetchall() == [("a",)]
