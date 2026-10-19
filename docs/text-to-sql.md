# Text to SQL

`kuhaku.tools.sql` answers natural-language questions over a database, with the same
access-control rule RAG applies to documents: whatever the caller is not entitled to
does not exist. The model is never shown it, SQL that names it is rejected exactly as if
it were not there, and rows outside the caller's row filter are never read.

SQLite works out of the box with the standard library. PostgreSQL needs an extra:

```bash
pip install "kuhaku[sql-postgres]"
```

## A first question

```python
import sqlite3
import tempfile
from pathlib import Path

from kuhaku import AuthContext
from kuhaku.tools.sql import AccessPolicy, RowFilter, SQLiteDatabase, TextToSQL


class CannedLLM:
    """Stands in for a real provider so this example runs offline. Leave ``llm`` out
    to use the provider configured in your environment, as RAG does."""

    name = "canned:demo"

    def generate(self, system: str, user: str) -> str:
        return '{"sql": "SELECT customer, total FROM orders ORDER BY total DESC", ' \
               '"explanation": "Orders, largest first."}'


with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "shop.db"
    with sqlite3.connect(path) as setup:
        setup.executescript(
            """
            CREATE TABLE orders (id INTEGER, tenant TEXT, customer TEXT, total REAL);
            CREATE TABLE payroll (employee TEXT, salary INTEGER);
            INSERT INTO orders VALUES (1, 'acme', 'Initech', 100), (2, 'acme', 'Umbrella', 250),
                                      (3, 'globex', 'Hooli', 999);
            """
        )
    setup.close()

    policy = AccessPolicy(
        tables={"payroll": ["people_ops"]},
        # No AuthContext means a NULL tenant, which matches no row: fail closed.
        row_filters={
            "orders": lambda auth: RowFilter(
                "tenant = :tenant", {"tenant": auth.metadata.get("tenant") if auth else None}
            )
        },
    )
    tool = TextToSQL(SQLiteDatabase(path), policy=policy, llm=CannedLLM())

    ada = AuthContext("ada", roles=("sales",), metadata={"tenant": "acme"})
    result = tool.ask("what are our biggest orders?", auth_context=ada)

    print(result.status)   # SQLStatus.ANSWERED
    print(result.columns)  # ('customer', 'total')
    print(result.rows)     # (('Umbrella', 250.0), ('Initech', 100.0)) -- no Hooli
    print(result.sql)      # the query that actually ran -- see "What runs" below
```

`result` is a `SQLResult`:

| field | meaning |
|---|---|
| `status` | `ANSWERED`, `NO_MATCH`, `REJECTED` or `FAILED` |
| `sql`, `params` | the validated query that ran (or would run), never the model's raw text |
| `columns`, `rows` | the result, at most `max_rows` rows; `truncated` says more existed |
| `explanation` | the model's one-line account of its query |
| `answer` | a natural-language answer, only with the optional `AnswerStep` |
| `issues` | why a non-`ANSWERED` result is what it is |
| `attempts`, `usage` | model calls made, and their summed token usage |
| `trace_id` | the id on every log record and span of this question |

`NO_MATCH` means the question cannot be answered from the schema the caller may see.
`REJECTED` means the model's SQL failed validation even after repair attempts.
`FAILED` means the model gave no usable reply or the database could not run the query.

## The access policy

```python
from kuhaku.tools.sql import AccessPolicy, RowFilter

policy = AccessPolicy(
    tables={"payroll": ["people_ops"]},                   # table-level tags
    columns={"customers.email": ["support", "people_ops"]},  # column-level tags
    row_filters={
        "tickets": RowFilter("archived = 0"),              # the same for everyone
        "orders": lambda auth: RowFilter(                  # per caller
            "region = :region",
            {"region": auth.metadata.get("region") if auth else None},
        ),
    },
)
```

Tags follow the [access-control](access-control.md) rule exactly:

- an **untagged** table or column is visible to everyone;
- a **tagged** one is visible only when one of its tags is among the caller's `roles`;
- a tagged one with **no `auth_context`** is invisible.

Flat, exact, case-sensitive set intersection. Two additions specific to tables:

- A column is visible only when its table is. A table left with **no** visible column
  is invisible too — `COUNT(*)` over it would still disclose its size.
- Every name in the policy is checked against the catalog when `TextToSQL` is built.
  A misspelt table name is a `ConfigError` then, not a table silently left open.

**Row filters** are SQL predicates over one table's columns, written by you, never by
the model — the only SQL the tool trusts. Values go in `params` and are bound as
parameters (`:name`), never spliced into the text. A filter may use a column the caller
cannot see (a hidden `tenant_id`, say). A callable source receives the caller's
`AuthContext` (or `None`) and returns a `RowFilter`, or `None` for no filter; if it
raises, the question fails. Make the `None` case fail closed, as above.

Wherever the model's SQL reads a filtered table — the main query, a join, a subquery, a
CTE, either side of a `UNION` — the table is replaced by
`(SELECT <visible columns> FROM table WHERE <filter>)`. An `OR 1 = 1` in the model's
`WHERE` lands outside that parenthesis and cannot widen it.

## What a blocked result looks like

The same as a missing one. A caller without `people_ops` who asks about salaries is
never shown `payroll`, so the model usually answers `NO_MATCH` — the same outcome as a
question about a table that does not exist. If the model guesses the name anyway, the
validator rejects `SELECT salary FROM payroll` with `unknown table 'payroll'`: the same
code and message, modulo the name, as `SELECT x FROM no_such_table`. The validator is
only ever given the entitled schema, so it could not tell the two apart if it tried, and
the repair feedback the model receives names only what its own SQL said.

The audit record does tell them apart: `metadata.denied_references` lists names that
exist but were not entitled, `metadata.unknown_references` names that do not exist.

## What runs

The model's SQL is never executed. The validator:

1. **Parses** it with an allow-list grammar for read-only `SELECT`: joins (inner, left,
   cross, comma), subqueries, non-recursive CTEs, `UNION`/`INTERSECT`/`EXCEPT`,
   `GROUP BY`/`HAVING`, `ORDER BY`, `LIMIT … OFFSET`, `CASE`, `CAST`, `IS`, `IN`,
   `BETWEEN`, `LIKE`, `EXISTS`, and an allow list of functions per dialect. Anything
   else is a rejection: every other statement, `;` between statements, comments, bind
   parameters, blob/hex literals, schema-qualified names, window functions,
   `RIGHT`/`FULL`/`NATURAL` joins, `USING`, `WITH RECURSIVE`, and every function not on
   the list (`load_extension`, `pg_read_file`, …).
2. **Resolves** every table and column reference — through aliases, subqueries, CTEs,
   correlated references and `*` — against the caller's entitled schema. Names compare
   without regard to case or quoting style, so `"SALARY"`, `[salary]` and `Salary` are
   the same column.
3. **Renders new SQL** from the resolved tree: every table gets a fresh alias, every
   column is written `"alias"."column"`, every CTE gets a fresh name, every `*` is
   expanded to the visible columns, every operator is parenthesized, string literals
   become bind parameters, row filters are applied, and the outermost `LIMIT` is capped
   at `max_rows`. How the database resolves a name can therefore never differ from how
   the validator checked it.

The validator is deterministic and can be used on its own:

```python
from kuhaku import AuthContext
from kuhaku.tools.sql import (
    AccessPolicy, Column, SchemaCatalog, SQLValidationError, Table, Validator,
    entitled_schema,
)

catalog = SchemaCatalog([
    Table("employees", (Column("name"), Column("salary"))),
])
policy = AccessPolicy(columns={"employees.salary": ["people_ops"]})
schema = entitled_schema(catalog, policy, AuthContext("ada", roles=("eng",)))
validator = Validator(schema, max_rows=50, reserved_names=catalog.names)

ok = validator.validate("SELECT * FROM employees WHERE name LIKE 'A%'")
print(ok.sql)     # SELECT "t1"."name" AS "name" FROM "employees" AS "t1" WHERE ("t1"."name" LIKE ?) LIMIT 50
print(ok.params)  # ('A%',)

for sql in ("SELECT salary FROM employees", "SELECT bonus FROM employees",
            "SELECT 1; DROP TABLE employees"):
    try:
        validator.validate(sql)
    except SQLValidationError as exc:
        print(exc.code, "-", exc)
# unknown_column - unknown column 'salary'
# unknown_column - unknown column 'bonus'
# multiple_statements - only a single statement is allowed
```

## Execution

`SQLiteDatabase(path)` opens the file **read-only** (`mode=ro`) for every query, and runs
each query under SQLite's authorizer, which allows only `SELECT`, reads of the tables
and columns the validator resolved (plus row-filter columns), and the dialect's allowed
functions. A query running past `timeout_seconds` (default 10) is interrupted; at most
`max_rows` (default 100) rows are returned. Lock contention is retried with backoff
through `kuhaku.core.retry`.

`SQLiteDatabase.from_connection(conn)` borrows an open connection — for an in-memory
database. The authorizer and timeout still apply, but the connection is not reopened
read-only, so the authorizer alone stands between a query and a write.

`PostgresDatabase(conninfo)` runs each query in a `READ ONLY` transaction with a
`statement_timeout`, then rolls back. PostgreSQL has no per-statement authorizer:
**connect as a role that can only `SELECT` the tables you catalog.**
`DBAPIDatabase(connect, dialect=..., tables=[...])` wraps any DB-API 2.0 driver;
read-only access and timeouts are then yours to configure (`setup=` runs per
connection).

A database error becomes `FAILED` with a generic message; the driver's own text goes to
the log, not the result.

## The catalog

`TextToSQL(database)` introspects the database. To describe it for the model, build the
catalog yourself:

```python
import sqlite3

from kuhaku.tools.sql import SchemaCatalog, SQLiteDatabase

connection = sqlite3.connect(":memory:")
connection.execute("CREATE TABLE orders (id INTEGER, region TEXT, total REAL)")
connection.execute("INSERT INTO orders VALUES (1, 'EU', 10), (2, 'US', 20)")
database = SQLiteDatabase.from_connection(connection)

catalog = SchemaCatalog.from_database(database, sample_values=5).describe(
    {"orders": "One row per customer order."},
    columns={"orders.total": "Order value in EUR, VAT included."},
)
print(catalog.get("orders").column("region").sample_values)  # ('EU', 'US')
```

`sample_values` (off by default — it reads data) shows the model a few distinct values
of each text column, so it writes `'EU'` rather than `'Europe'`. They are **never shown
for a table with a row filter**: they are read without any filter, and could show one
tenant another tenant's values. `from_database(database, tables=[...])` limits the
catalog to the named tables; the rest of the database is then unreachable.

## The pipeline

Like extraction, a question runs through named steps:

| step | | |
|---|---|---|
| `schema` | protected | the caller's entitled schema, and the (optionally pruned) part the model sees |
| `generate` | required | the model call, the repair loop, fallbacks |
| `validate` | protected | the authoritative validation; recomputes entitlement, trusts nothing upstream |
| `execute` | optional | added when there is a database |
| `answer` | optional | `AnswerStep()`: a short natural-language answer from the rows |

`schema` and `validate` are the access-control boundary: they cannot be removed,
disabled or replaced. `generate` can be replaced — whatever it produces still goes
through `validate`.

```python
from kuhaku.tools.sql import AnswerStep, Column, SchemaCatalog, Table, TextToSQL


class CannedLLM:
    name = "canned:demo"

    def generate(self, system: str, user: str) -> str:
        return '{"sql": "SELECT name FROM teams", "explanation": "All teams."}'


catalog = SchemaCatalog([Table("teams", (Column("name"),))])
tool = TextToSQL(catalog=catalog, llm=CannedLLM(), max_tables=5, steps=[AnswerStep()])
print(tool.pipeline)  # Pipeline([schema, generate, validate, answer])
```

Without a `database`, `TextToSQL(catalog=...)` validates SQL without running it.

Other options: `max_repair_attempts` (default 2) bounds the repair loop;
`fallback_llms` are tried in order when a provider is unreachable or its SQL stays
unusable (`NO_MATCH` never escalates — a stronger model shown the same schema cannot
find what is not in it); `native_structured_output` uses the provider's native JSON
schema mode when it has one; `max_tables` ranks tables by word overlap with the
question and shows the model only the top ones — pruning narrows the prompt, never what
the validator accepts. `SchemaStep(pruner=...)` takes your own ranking function; it is
only ever given the entitled tables.

## Audit

Every question writes one record through `kuhaku.core.security.audit`, at
`Settings.audit_log_path`, unless `Settings.audit_enabled` is `False`. It carries the
identity and roles, a hash of the question and its PII-sanitized prefix, the status,
the tables and columns read, a hash of the executed SQL, the row count, and the
denied/unknown references. **Never row data, and never the SQL text** — its literals
come from the question.

## What is and is not guaranteed

Guaranteed, by the validator and covered by the test suite:

- the model is shown only entitled tables and columns;
- SQL that references a non-entitled table or column is rejected with the same code and
  message as a nonexistent one, in every position (select list, `WHERE`, joins,
  subqueries, CTEs, `UNION` branches, `ORDER BY`, `HAVING`, `*`);
- only a single read-only `SELECT` in the accepted subset runs, rendered by the
  validator;
- row filters apply to every read of their table;
- the outermost `LIMIT` never exceeds `max_rows`.

Not guaranteed:

- **Inference.** A caller can learn about rows they may read through aggregates over
  them; row filters bound what is read, not what can be deduced from it.
- **Timing.** A rejection is not constant-time.
- **Your row filters' correctness.** They are trusted SQL; a filter that matches too
  much is not caught.
- **PostgreSQL and DB-API backends** have no second, database-side check of which
  columns are read; use a least-privilege role.
- The answer step sends result rows to the model — rows the caller was entitled to.
