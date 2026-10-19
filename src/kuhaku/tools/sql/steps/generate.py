"""The core step: ask the model for SQL, and repair it until it validates."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from kuhaku.core.llm.base import (
    LLMError,
    LLMProvider,
    LLMUnavailableError,
    StructuredOutputError,
)
from kuhaku.core.llm.provider import as_chat_provider, capabilities_of
from kuhaku.core.models import Message

from ..context import SQLContext
from ..pipeline import GENERATE_STEP, Step
from ..prompts import (
    RESPONSE_SCHEMA,
    build_repair_prompt,
    build_system_prompt,
    build_user_prompt,
)
from ..result import Issue, IssueKind
from ..validator import SQLValidationError, Validator
from .schema import NO_MATCH_MESSAGE

logger = logging.getLogger(__name__)

_SQL_FENCE = re.compile(r"```(?:sql)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class ReplyParseError(ValueError):
    """The reply is neither the JSON object asked for nor a fenced SQL block."""


class GenerateStep(Step):
    """Calls the model for ``{"sql": ..., "explanation": ...}`` and checks the SQL with
    the same deterministic :class:`~kuhaku.tools.sql.validator.Validator` the
    ``validate`` step uses. A rejection is fed back to the model -- the validator's
    message, which only ever names what the SQL itself said -- for up to
    ``max_repair_attempts`` further calls.

    ``"sql": null`` means the model found the question unanswerable from the schema it
    was shown; that is ``NO_MATCH`` and is neither repaired nor escalated, since a
    stronger model shown the same schema cannot find what is not in it.

    ``fallbacks`` and native structured output behave as in the extraction tool's
    ``ExtractStep``: a provider that cannot be reached or whose output stays unusable
    hands over to the next; a failure on the last raises
    :class:`~kuhaku.core.llm.base.LLMUnavailableError`.
    """

    name = GENERATE_STEP

    def __init__(
        self,
        *,
        max_repair_attempts: int = 2,
        fallbacks: Sequence[LLMProvider] = (),
        native_structured_output: bool = True,
        instructions: Sequence[str] = (),
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")
        self.max_repair_attempts = max_repair_attempts
        self.fallbacks = list(fallbacks)
        self.native_structured_output = native_structured_output
        self.instructions = list(instructions)

    def run(self, ctx: SQLContext) -> None:
        assert ctx.schema is not None and ctx.prompt_schema is not None
        validator = Validator(
            ctx.schema,
            dialect=ctx.dialect,
            max_rows=ctx.max_rows,
            reserved_names=ctx.catalog.names,
        )
        system = build_system_prompt(ctx.dialect, self.instructions)
        user = build_user_prompt(ctx.question, ctx.prompt_schema)

        providers = [ctx.llm, *self.fallbacks]
        unresolved: list[Issue] = []
        for position, llm in enumerate(providers):
            is_last = position == len(providers) - 1
            try:
                unresolved = self._attempt(ctx, llm, validator, system, user)
            except LLMError as exc:
                if is_last:
                    raise LLMUnavailableError(str(exc)) from exc
                logger.warning(
                    "sql provider unavailable, falling back",
                    extra={"llm": llm.name, "error": str(exc)},
                )
                continue
            if not unresolved or unresolved[0].kind is IssueKind.NO_MATCH:
                ctx.metadata["llm"] = llm.name
                ctx.metadata["structured_output"] = "native" if self._native(llm) else "prompt"
                break
            if not is_last:
                logger.info(
                    "sql output unusable, escalating",
                    extra={"llm": llm.name, "next_llm": providers[position + 1].name},
                )
        if unresolved:
            ctx.issues.extend(unresolved)
            ctx.candidate_sql = None
            ctx.stop()

    def _attempt(
        self,
        ctx: SQLContext,
        llm: LLMProvider,
        validator: Validator,
        system: str,
        user: str,
    ) -> list[Issue]:
        prompt = user
        unresolved: list[Issue] = []
        for _ in range(1 + self.max_repair_attempts):
            ctx.attempts += 1
            raw = self._complete(ctx, llm, system, prompt)
            try:
                sql, explanation = parse_reply(raw)
            except ReplyParseError as exc:
                unresolved = [Issue(IssueKind.FAILED, "parse_error", str(exc))]
                prompt = build_repair_prompt(user, raw, [str(exc)])
                continue
            ctx.explanation = explanation
            if sql is None:
                return [Issue(IssueKind.NO_MATCH, "no_match", NO_MATCH_MESSAGE)]
            try:
                validator.validate(sql)
            except SQLValidationError as exc:
                if exc.reference:
                    ctx.rejected_references.append(exc.reference)
                unresolved = [Issue(IssueKind.REJECTED, exc.code, str(exc))]
                prompt = build_repair_prompt(user, raw, [str(exc)])
                continue
            ctx.candidate_sql = sql
            return []
        return unresolved

    def _native(self, llm: LLMProvider) -> bool:
        return self.native_structured_output and capabilities_of(llm).native_structured_output

    def _complete(self, ctx: SQLContext, llm: LLMProvider, system: str, prompt: str) -> str:
        messages = [Message.system(system), Message.user(prompt)]
        chat = as_chat_provider(llm)
        if not self._native(llm):
            response = chat.chat(messages)
            ctx.add_usage(response.usage)
            return response.text
        try:
            structured = chat.generate_structured(
                messages,
                RESPONSE_SCHEMA,
                name="sql_query",
                # This step's own loop does the repairing, with its own budget.
                max_repair_attempts=0,
            )
        except StructuredOutputError as exc:
            # Unparseable even in native mode: take the ordinary parse-error repair path.
            return exc.raw
        ctx.add_usage(structured.usage)
        return structured.text


def parse_reply(raw: str) -> tuple[str | None, str]:
    """``(sql or None, explanation)`` from the model's reply.

    Accepts the JSON object asked for, tolerating a code fence or prose around it, and
    as a last resort a fenced ```sql block -- providers without a native JSON mode
    produce one often enough that refusing it would spend a repair on formatting.
    """

    text = raw.strip()
    parsed = _json_object(text)
    if parsed is None:
        fenced = _SQL_FENCE.search(raw)
        if fenced:
            return fenced.group(1).strip(), ""
        raise ReplyParseError("the reply is not a JSON object with \"sql\" and \"explanation\"")
    if "sql" not in parsed:
        raise ReplyParseError('the reply has no "sql" key')
    sql = parsed["sql"]
    if sql is not None and not isinstance(sql, str):
        raise ReplyParseError('"sql" must be a string or null')
    if isinstance(sql, str) and not sql.strip():
        sql = None
    explanation = parsed.get("explanation")
    return sql, explanation if isinstance(explanation, str) else ""


def _json_object(text: str) -> dict[str, Any] | None:
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
