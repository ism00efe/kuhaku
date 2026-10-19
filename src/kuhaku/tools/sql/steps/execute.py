"""Optional step: run the validated query."""

from __future__ import annotations

import logging

from kuhaku.core.exceptions import ConfigError

from ..context import SQLContext
from ..database import Database, QueryExecutionError
from ..pipeline import EXECUTE_STEP, Step
from ..result import Issue, IssueKind

logger = logging.getLogger(__name__)


class ExecuteStep(Step):
    """Runs ``ctx.validated`` -- never anything else -- with a time budget and the row
    cap. A database error becomes a ``FAILED`` issue with a generic message: the
    driver's own text can name objects, so it goes to the log, not the result."""

    name = EXECUTE_STEP

    def __init__(self, database: Database, *, timeout_seconds: float = 10.0) -> None:
        if timeout_seconds <= 0:
            raise ConfigError("timeout_seconds must be positive.")
        self.database = database
        self.timeout_seconds = timeout_seconds

    def run(self, ctx: SQLContext) -> None:
        if ctx.validated is None:
            return
        try:
            ctx.rows = self.database.execute(
                ctx.validated, max_rows=ctx.max_rows, timeout_seconds=self.timeout_seconds
            )
        except QueryExecutionError as exc:
            logger.warning(
                "sql execution failed",
                extra={"timed_out": exc.timed_out, "error": str(exc)},
            )
            if exc.timed_out:
                issue = Issue(
                    IssueKind.FAILED,
                    "timeout",
                    f"the query ran longer than {self.timeout_seconds:g}s and was stopped",
                )
            else:
                issue = Issue(
                    IssueKind.FAILED, "execution_error", "the database could not run the query"
                )
            ctx.issues.append(issue)
            ctx.stop()
            return
        ctx.metadata["row_count"] = len(ctx.rows.rows)
