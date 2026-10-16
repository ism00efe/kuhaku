"""The core step: ask the model, parse its reply, validate it against the schema."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from kuhaku.core.llm.base import LLMError, LLMProvider, LLMUnavailableError

from ..context import ExtractionContext
from ..parsing import OutputParseError, parse_envelope, validate_data
from ..pipeline import EXTRACT_STEP, Step
from ..prompts import build_retry_prompt, build_system_prompt, build_user_prompt
from ..result import Issue, IssueKind

logger = logging.getLogger(__name__)


@dataclass
class _Reply:
    """One provider's last reply and what came of it. Kept apart per provider so a
    result never mixes one provider's data with another's raw output."""

    llm: str
    raw: str
    response: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    value: BaseModel | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not any(issue.kind is IssueKind.RETRYABLE for issue in self.issues)


class ExtractStep(Step):
    """Calls the LLM and turns its reply into a validated instance of the schema model.

    Malformed JSON and schema violations are fed back to the model for up to
    ``max_retries`` further calls. Values the text does not contain are not
    retried -- asking again cannot make them appear -- and are reported as missing.

    ``fallbacks`` are further providers tried in order, each with its own retry
    budget, when the one before could not produce usable output or could not be
    reached -- a cheap model first, a stronger one only for the texts it gets wrong.
    Missing values never fall back: a stronger model cannot find what the text lacks.

    The result is built from exactly one provider's reply: the first usable one, else
    the last one that replied at all (preferring one that had data). Its provider is
    recorded as ``metadata["llm"]``. :class:`LLMUnavailableError` is raised only when
    no provider replied -- a provider failure says nothing about the text, and the
    caller's retry policy for an unreachable service is not this step's to decide.
    """

    name = EXTRACT_STEP

    def __init__(
        self, *, max_retries: int = 1, fallbacks: Sequence[LLMProvider] = ()
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self.fallbacks = list(fallbacks)

    @staticmethod
    def build_prompts(ctx: ExtractionContext) -> tuple[str, str]:
        """``(system, user)`` for ``ctx`` after the prepare phase -- what :meth:`run`
        sends first, and what :meth:`Extractor.preview` shows."""

        system = build_system_prompt(ctx.schema, ctx.instructions, ctx.prompts)
        user = build_user_prompt(
            schema=ctx.schema,
            text=ctx.text,
            response_fields=ctx.response_fields,
            history=ctx.history,
            reference=ctx.reference,
            blocks=ctx.prompt_blocks,
        )
        return system, user

    def run(self, ctx: ExtractionContext) -> None:
        if not ctx.text.strip():
            # An empty text contains no values -- what the model would answer, known
            # without paying for a call.
            data: dict[str, Any] = {
                field.alias or name: None for name, field in ctx.schema.model.model_fields.items()
            }
            value, missing, errors = validate_data(ctx.schema.model, data)
            ctx.response, ctx.data, ctx.value = {"data": data}, data, value
            ctx.issues.extend(errors + missing)
            return

        system, user = self.build_prompts(ctx)

        providers = [ctx.llm, *self.fallbacks]
        replies: list[_Reply] = []
        last_error: LLMError | None = None
        for position, llm in enumerate(providers):
            try:
                reply = self._attempt(ctx, llm, system, user)
            except LLMError as exc:
                last_error = exc
                logger.warning(
                    "extraction provider unavailable", extra={"llm": llm.name, "error": str(exc)}
                )
                continue
            replies.append(reply)
            if reply.usable:
                break
            if position < len(providers) - 1:
                logger.info(
                    "extraction output unusable, falling back",
                    extra={"llm": llm.name, "next_llm": providers[position + 1].name},
                )

        if not replies:
            raise LLMUnavailableError(str(last_error)) from last_error
        self._commit(ctx, _choose(replies))

    def _attempt(
        self, ctx: ExtractionContext, llm: LLMProvider, system: str, user: str
    ) -> _Reply:
        """One provider's attempt plus its retries."""

        prompt = user
        reply = _Reply(llm=llm.name, raw="")
        for _ in range(1 + self.max_retries):
            ctx.attempts += 1
            raw = llm.generate(system, prompt)
            if not isinstance(raw, str):
                # A provider (often a hand-written one) that returns None or a non-str
                # must not crash parsing; it is an unusable reply like any other.
                problem = f"the reply is not text (got {type(raw).__name__})"
                reply = _Reply(
                    llm=llm.name,
                    raw=repr(raw),
                    issues=[Issue(IssueKind.RETRYABLE, "parse_error", problem)],
                )
                prompt = build_retry_prompt(user, reply.raw, [problem], ctx.prompts)
                continue

            try:
                envelope = parse_envelope(raw)
            except OutputParseError as exc:
                reply = _Reply(
                    llm=llm.name,
                    raw=raw,
                    issues=[Issue(IssueKind.RETRYABLE, "parse_error", str(exc))],
                )
                prompt = build_retry_prompt(user, raw, [str(exc)], ctx.prompts)
                continue

            value, missing, errors = validate_data(ctx.schema.model, envelope["data"])
            reply = _Reply(
                llm=llm.name,
                raw=raw,
                response=envelope,
                data=envelope["data"],
                value=value,
                issues=errors + missing,
            )
            if not errors:
                return reply
            prompt = build_retry_prompt(
                user, raw, [issue.message for issue in errors], ctx.prompts
            )
        return reply

    @staticmethod
    def _commit(ctx: ExtractionContext, reply: _Reply) -> None:
        ctx.raw_output = reply.raw
        ctx.response = reply.response
        ctx.data = reply.data
        ctx.value = reply.value if reply.usable else None
        ctx.issues.extend(reply.issues)
        ctx.metadata["llm"] = reply.llm
        # With no data at all there is nothing for later steps to check. With data that
        # still breaks the schema, later steps (evidence, say) can still report on it.
        if ctx.data is None:
            ctx.stop()


def _choose(replies: list[_Reply]) -> _Reply:
    for reply in replies:
        if reply.usable:
            return reply
    with_data = [reply for reply in replies if reply.data is not None]
    return (with_data or replies)[-1]
