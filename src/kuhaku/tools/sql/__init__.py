"""Access-controlled natural language -> SQL, as a third kuhaku tool.

Built only on ``kuhaku.core`` (LLM abstraction, auth, audit, observability, retry) and
the standard library; it imports nothing from another tool. SQLite works out of the
box; PostgreSQL needs the ``sql-postgres`` extra.

    from kuhaku import AuthContext
    from kuhaku.tools.sql import AccessPolicy, RowFilter, SQLiteDatabase, TextToSQL

    policy = AccessPolicy(
        tables={"salaries": ["hr"]},
        columns={"employees.email": ["hr"]},
        # No AuthContext -> tenant NULL -> no rows: a row filter should fail closed.
        row_filters={"orders": lambda auth: RowFilter(
            "tenant_id = :tenant", {"tenant": auth.metadata.get("tenant") if auth else None})},
    )
    tool = TextToSQL(SQLiteDatabase("shop.db"), policy=policy)
    result = tool.ask("top 5 customers by revenue", auth_context=AuthContext("ada"))
    result.status, result.sql, result.columns, result.rows

The model is shown only the tables and columns the caller is entitled to. Its SQL is
parsed with an allow-list grammar, every name is resolved against that same entitled
schema, and the query that runs is re-rendered from the resolved tree with row
filters, bind parameters and a row cap applied -- see :mod:`.validator`.
"""

from .access import (
    AccessPolicy,
    EntitledSchema,
    EntitledTable,
    RowFilter,
    RowFilterSource,
    entitled_schema,
    is_entitled,
)
from .catalog import Column, SchemaCatalog, Table
from .context import SQLContext
from .database import (
    Database,
    DBAPIDatabase,
    QueryExecutionError,
    QueryRows,
    SQLiteDatabase,
)
from .dialect import POSTGRES, SQLITE, Dialect
from .engine import TextToSQL
from .parser import SQLSyntaxError
from .pipeline import (
    ANSWER_STEP,
    EXECUTE_STEP,
    GENERATE_STEP,
    SCHEMA_STEP,
    VALIDATE_STEP,
    Pipeline,
    Step,
)
from .postgres import PostgresDatabase
from .result import Issue, IssueKind, SQLResult, SQLStatus
from .steps import (
    AnswerStep,
    ExecuteStep,
    GenerateStep,
    Pruner,
    SchemaStep,
    ValidateStep,
)
from .validator import SQLValidationError, ValidatedQuery, Validator

__all__ = [
    "ANSWER_STEP",
    "EXECUTE_STEP",
    "GENERATE_STEP",
    "POSTGRES",
    "SCHEMA_STEP",
    "SQLITE",
    "VALIDATE_STEP",
    "AccessPolicy",
    "AnswerStep",
    "Column",
    "DBAPIDatabase",
    "Database",
    "Dialect",
    "EntitledSchema",
    "EntitledTable",
    "ExecuteStep",
    "GenerateStep",
    "Issue",
    "IssueKind",
    "Pipeline",
    "PostgresDatabase",
    "Pruner",
    "QueryExecutionError",
    "QueryRows",
    "RowFilter",
    "RowFilterSource",
    "SQLContext",
    "SQLResult",
    "SQLStatus",
    "SQLSyntaxError",
    "SQLValidationError",
    "SQLiteDatabase",
    "SchemaCatalog",
    "SchemaStep",
    "Step",
    "Table",
    "TextToSQL",
    "ValidateStep",
    "ValidatedQuery",
    "Validator",
    "entitled_schema",
    "is_entitled",
]
