"""Optional step: a short natural-language answer from the result rows."""

from __future__ import annotations

import logging

from kuhaku.core.llm.base import LLMError
from kuhaku.core.llm.provider import as_chat_provider
from kuhaku.core.models import Message

from ..context import SQLContext
from ..pipeline import ANSWER_STEP, Step
from ..prompts import ANSWER_RULES, build_answer_prompt

logger = logging.getLogger(__name__)


class AnswerStep(Step):
    """Asks the model to answer the question from the rows. Runs only after the
    execute step produced rows. The model sees at most ``max_rows_in_prompt`` of them --
    rows the caller was already entitled to, since they come from the validated query.

    A provider failure here leaves the rest of the result intact (``answer`` stays
    ``None`` and ``metadata["answer_error"]`` says why): the query already ran, and
    discarding its rows over a summary would lose the part that worked.
    """

    name = ANSWER_STEP

    def __init__(self, *, max_rows_in_prompt: int = 50) -> None:
        self.max_rows_in_prompt = max_rows_in_prompt

    def run(self, ctx: SQLContext) -> None:
        if ctx.rows is None:
            return
        rows = ctx.rows.rows[: self.max_rows_in_prompt]
        truncated = ctx.rows.truncated or len(ctx.rows.rows) > len(rows)
        prompt = build_answer_prompt(ctx.question, ctx.rows.columns, rows, truncated)
        try:
            response = as_chat_provider(ctx.llm).chat(
                [Message.system(ANSWER_RULES), Message.user(prompt)]
            )
        except LLMError as exc:
            logger.warning("sql answer step failed", extra={"error": str(exc)})
            ctx.metadata["answer_error"] = str(exc)
            return
        ctx.add_usage(response.usage)
        ctx.answer = response.text.strip()
