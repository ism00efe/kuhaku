"""Protected step: the authoritative validation of whatever SQL the generate step left.

It does not trust the generate step -- which can be replaced -- to have validated
anything: it recomputes the caller's entitlement from the catalog and the policy, and
validates the candidate from scratch. Only its output, ``ctx.validated``, is ever
executed.
"""

from __future__ import annotations

from ..access import entitled_schema
from ..context import SQLContext
from ..pipeline import VALIDATE_STEP, Step
from ..result import Issue, IssueKind
from ..validator import SQLValidationError, Validator


class ValidateStep(Step):
    name = VALIDATE_STEP

    def run(self, ctx: SQLContext) -> None:
        if ctx.candidate_sql is None:
            ctx.issues.append(
                Issue(IssueKind.FAILED, "no_sql", "the generate step produced no SQL")
            )
            ctx.stop()
            return
        validator = Validator(
            entitled_schema(ctx.catalog, ctx.policy, ctx.auth_context),
            dialect=ctx.dialect,
            max_rows=ctx.max_rows,
            reserved_names=ctx.catalog.names,
        )
        try:
            ctx.validated = validator.validate(ctx.candidate_sql)
        except SQLValidationError as exc:
            if exc.reference:
                ctx.rejected_references.append(exc.reference)
            ctx.issues.append(Issue(IssueKind.REJECTED, exc.code, str(exc)))
            ctx.stop()
