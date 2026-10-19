"""``TextToSQL``: the entry point -- one catalog, one access policy, one pipeline."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

from kuhaku.core.auth import AuthContext
from kuhaku.core.config import Settings, get_settings
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.base import LLMProvider
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.observability import bind_trace_id, instrumented_step
from kuhaku.core.sanitization import sanitize
from kuhaku.core.security.audit import AuditWriteError
from kuhaku.core.security.audit import record as record_audit

from .access import AccessPolicy
from .catalog import SchemaCatalog, split_column_ref
from .context import SQLContext
from .database import Database
from .dialect import SQLITE, Dialect
from .pipeline import Pipeline, Step
from .result import SQLResult
from .steps import ExecuteStep, GenerateStep, SchemaStep, ValidateStep

logger = logging.getLogger(__name__)

AUDIT_EVENT = "sql_query"


class TextToSQL:
    """Answers questions over a database, showing the model and executing only what
    the caller is entitled to.

    ``database`` is introspected into the catalog unless ``catalog`` is given, and
    queries are executed against it (the ``execute`` step). Without a ``database``,
    pass a ``catalog``: the tool then validates SQL without running it.

    ``policy`` holds the table/column access tags and row filters; every name in it is
    checked against the catalog here, so a typo fails now rather than leaving a table
    unprotected. ``max_rows`` caps the outermost ``LIMIT`` and the rows returned;
    ``timeout_seconds`` bounds each execution.

    ``max_tables`` turns on relevance pruning of the schema shown to the model;
    ``steps`` are appended after the default ones (``schema``, ``generate``,
    ``validate``, and ``execute`` when there is a database) -- e.g.
    ``steps=[AnswerStep()]``. :attr:`pipeline` composes further.

    ``llm``, ``fallback_llms``, ``enable_token_tracking`` and
    ``native_structured_output`` mean what they mean for
    :class:`~kuhaku.tools.extraction.Extractor`. ``settings`` supplies the provider when
    ``llm`` is omitted and the audit-log location (``audit_enabled``,
    ``audit_log_path``).

    Every question writes one audit record -- identity, roles, status, tables and
    columns read, a hash of the SQL, row count, and any names the SQL used that did not
    resolve. Never row data, never the SQL text (it carries the question's literals).
    """

    def __init__(
        self,
        database: Database | None = None,
        *,
        catalog: SchemaCatalog | None = None,
        policy: AccessPolicy | None = None,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
        dialect: Dialect | None = None,
        steps: Sequence[Step] = (),
        max_rows: int = 100,
        timeout_seconds: float = 10.0,
        max_repair_attempts: int = 2,
        max_tables: int | None = None,
        fallback_llms: Sequence[LLMProvider] = (),
        enable_token_tracking: bool = True,
        native_structured_output: bool = True,
    ) -> None:
        if database is None and catalog is None:
            raise ConfigError("Pass a database, a catalog, or both.")
        if max_rows < 1:
            raise ConfigError("max_rows must be at least 1.")
        self.database = database
        self.catalog = catalog if catalog is not None else SchemaCatalog.from_database(database)  # type: ignore[arg-type]
        self.policy = policy or AccessPolicy()
        self.policy.check(self.catalog)
        if dialect is not None:
            self.dialect = dialect
        elif database is not None:
            self.dialect = database.dialect
        else:
            self.dialect = SQLITE
        self.max_rows = max_rows
        self.settings = settings or get_settings()

        built = llm if llm is not None else build_llm_provider(self.settings)
        track = _token_tracked if enable_token_tracking else _untracked
        self.llm: LLMProvider = track(built)
        fallbacks = [track(fallback) for fallback in fallback_llms]

        default_steps: list[Step] = [
            SchemaStep(max_tables=max_tables),
            GenerateStep(
                max_repair_attempts=max_repair_attempts,
                fallbacks=fallbacks,
                native_structured_output=native_structured_output,
            ),
            ValidateStep(),
        ]
        if database is not None:
            default_steps.append(ExecuteStep(database, timeout_seconds=timeout_seconds))
        self.pipeline = Pipeline([*default_steps, *steps])

    def ask(
        self,
        question: str,
        *,
        auth_context: AuthContext | None = None,
        trace_id: str | None = None,
    ) -> SQLResult:
        """Answer ``question`` as ``auth_context``. ``None`` sees only untagged tables
        and columns, as in RAG.

        ``trace_id`` binds every log record and span of this call to an id; without
        one, an id bound by the caller is inherited, else a new one is minted.
        """

        if not isinstance(question, str):
            raise TypeError(f"question must be a str, got {type(question).__name__}.")
        with (
            bind_trace_id(trace_id) as tid,
            instrumented_step("sql", logger) as rec,
        ):
            ctx = SQLContext(
                question=question,
                auth_context=auth_context,
                llm=self.llm,
                catalog=self.catalog,
                policy=self.policy,
                dialect=self.dialect,
                max_rows=self.max_rows,
                trace_id=tid,
            )
            try:
                self.pipeline.run(ctx)
            except Exception:
                # A provider failure raises rather than becoming a result; it is still
                # a request someone made, so it is still audited.
                self._audit(ctx, "error")
                raise
            result = ctx.to_result()
            self._audit(ctx, result.status.value)
            rec.set(outcome=result.status.value, attempts=result.attempts)
            return result

    def _audit(self, ctx: SQLContext, status: str) -> None:
        validated = ctx.validated
        metadata: dict[str, object] = {
            "tool": "sql",
            "status": status,
            "issue_codes": [issue.code for issue in ctx.issues],
            "tables": list(validated.tables) if validated else [],
            "columns": list(validated.columns) if validated else [],
            "sql_sha256": (
                hashlib.sha256(validated.sql.encode("utf-8")).hexdigest() if validated else None
            ),
            "row_count": len(ctx.rows.rows) if ctx.rows is not None else None,
            "attempts": ctx.attempts,
            # Names the model used that did not resolve, split by whether they exist.
            # The caller cannot tell the two apart; the operator must be able to.
            "denied_references": sorted(
                {ref for ref in ctx.rejected_references if self._exists(ref)}
            ),
            "unknown_references": sorted(
                {ref for ref in ctx.rejected_references if not self._exists(ref)}
            ),
        }
        try:
            record_audit(
                self.settings.audit_log_path,
                enabled=self.settings.audit_enabled,
                trace_id=ctx.trace_id,
                raw_question=ctx.question,
                sanitized_retrieval_query=sanitize(ctx.question),
                event_type=AUDIT_EVENT,
                action=status,
                auth_context=ctx.auth_context,
                metadata=metadata,
            )
        except AuditWriteError:
            # As in RAGEngine: the record is for the operator, and a broken sink is
            # already logged and counted by record_audit itself.
            logger.exception(
                "audit record could not be written; result unaffected",
                extra={"event_type": AUDIT_EVENT, "trace_id": ctx.trace_id},
            )

    def _exists(self, reference: str) -> bool:
        if reference in self.catalog:
            return True
        if "." in reference:
            try:
                table_name, column_name = split_column_ref(reference)
            except ConfigError:
                return False
            table = self.catalog.get(table_name)
            if table is not None:
                return table.column(column_name) is not None
            # An alias qualifier: fall through to "any table has this column".
            reference = column_name
        return any(table.column(reference) is not None for table in self.catalog.tables)


def _token_tracked(llm: LLMProvider) -> LLMProvider:
    if isinstance(llm, TokenTrackingLLM):
        return llm
    return TokenTrackingLLM(llm, provider=llm.name.split(":", 1)[0])


def _untracked(llm: LLMProvider) -> LLMProvider:
    return llm
