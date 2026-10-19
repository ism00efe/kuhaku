"""The state one question carries through the pipeline. Steps communicate only through
this object, as in the extraction tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kuhaku.core.auth import AuthContext
from kuhaku.core.llm.base import LLMProvider, TokenUsage, combine_usage

from .access import AccessPolicy, EntitledSchema
from .catalog import SchemaCatalog
from .database import QueryRows
from .dialect import Dialect
from .result import Issue, SQLResult, SQLStatus, status_from_issues
from .validator import ValidatedQuery


@dataclass
class SQLContext:
    question: str
    auth_context: AuthContext | None
    llm: LLMProvider
    catalog: SchemaCatalog
    policy: AccessPolicy
    dialect: Dialect
    max_rows: int
    trace_id: str = ""

    # Written by the schema step. ``schema`` is everything the caller is entitled to;
    # ``prompt_schema`` the (possibly pruned) part of it the model is shown.
    schema: EntitledSchema | None = None
    prompt_schema: EntitledSchema | None = None

    # Written by the generate step.
    candidate_sql: str | None = None
    explanation: str = ""

    # Written by the validate step -- the only thing the execute step will run.
    validated: ValidatedQuery | None = None

    # Written by the execute and answer steps.
    rows: QueryRows | None = None
    answer: str | None = None

    issues: list[Issue] = field(default_factory=list)
    rejected_references: list[str] = field(default_factory=list)
    """Names the SQL used that did not resolve -- for the audit record only."""

    attempts: int = 0
    usage: TokenUsage | None = None
    stopped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def stop(self) -> None:
        self.stopped = True

    def add_usage(self, usage: TokenUsage | None) -> None:
        self.usage = combine_usage(self.usage, usage)

    def to_result(self) -> SQLResult:
        status = status_from_issues(self.issues)
        answered = status is SQLStatus.ANSWERED and self.validated is not None
        rows = self.rows
        truncated = False
        if rows is not None and self.validated is not None:
            truncated = rows.truncated or (
                self.validated.limit_capped and len(rows.rows) >= self.max_rows
            )
        return SQLResult(
            status=status,
            question=self.question,
            sql=self.validated.display_sql if answered and self.validated else None,
            params=self.validated.display_params if answered and self.validated else (),
            columns=rows.columns if answered and rows else (),
            rows=rows.rows if answered and rows else (),
            truncated=truncated if answered else False,
            explanation=self.explanation,
            answer=self.answer if answered else None,
            issues=tuple(self.issues),
            attempts=self.attempts,
            usage=self.usage,
            metadata=dict(self.metadata),
            trace_id=self.trace_id,
        )
