"""The core step: ask the model, parse its reply, validate it against the schema."""

from __future__ import annotations

from kuhaku.core.llm.base import LLMError, LLMUnavailableError

from ..context import ExtractionContext
from ..parsing import OutputParseError, parse_envelope, validate_data
from ..pipeline import EXTRACT_STEP, Step
from ..prompts import build_repair_prompt, build_system_prompt, build_user_prompt
from ..result import Issue, IssueKind


class ExtractStep(Step):
    """Calls the LLM and turns its reply into a validated instance of the schema model.

    Malformed JSON and schema violations are fed back to the model for up to
    ``max_repair_attempts`` further calls. Values the text does not contain are not
    retried -- asking again cannot make them appear -- and are reported as missing.

    A provider failure raises :class:`LLMUnavailableError` rather than becoming a
    result: it says nothing about the text, and the caller's retry/backoff policy
    for an unreachable service is not this step's to decide.
    """

    name = EXTRACT_STEP

    def __init__(self, *, max_repair_attempts: int = 1) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")
        self.max_repair_attempts = max_repair_attempts

    def run(self, ctx: ExtractionContext) -> None:
        system = build_system_prompt(ctx.schema, ctx.instructions)
        user = build_user_prompt(
            schema=ctx.schema,
            text=ctx.text,
            response_fields=ctx.response_fields,
            history=ctx.history,
            reference=ctx.reference,
            blocks=ctx.prompt_blocks,
        )

        prompt = user
        unresolved: list[Issue] = []
        for _ in range(1 + self.max_repair_attempts):
            ctx.attempts += 1
            try:
                raw = ctx.llm.generate(system, prompt)
            except LLMError as exc:
                raise LLMUnavailableError(str(exc)) from exc
            ctx.raw_output = raw

            try:
                envelope = parse_envelope(raw)
            except OutputParseError as exc:
                unresolved = [Issue(IssueKind.REPAIRABLE, "parse_error", str(exc))]
                prompt = build_repair_prompt(user, raw, [str(exc)])
                continue

            ctx.response = envelope
            ctx.data = envelope["data"]
            value, missing, errors = validate_data(ctx.schema.model, ctx.data)
            if not errors:
                ctx.value = value
                ctx.issues.extend(missing)
                return
            unresolved = errors + missing
            prompt = build_repair_prompt(user, raw, [issue.message for issue in errors])

        ctx.issues.extend(unresolved)
        # With no data at all there is nothing for later steps to check. With data that
        # still breaks the schema, later steps (evidence, say) can still report on it.
        if ctx.data is None:
            ctx.stop()
