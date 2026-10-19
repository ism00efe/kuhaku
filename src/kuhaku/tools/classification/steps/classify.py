"""The core step: ask the model, check its answer against the label set, decide or abstain."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from kuhaku.core.llm.base import LLMError, LLMProvider
from kuhaku.core.observability import start_span
from kuhaku.core.prompts import (
    PromptRef,
    PromptTemplate,
    prompt_metadata,
    record_prompts,
)

from ..context import ClassificationContext
from ..escalation import Attempt, capable_providers, escalate, uses_native
from ..parsing import OutputParseError, Reply, parse_reply
from ..pipeline import CLASSIFY_STEP, Step
from ..prompts import (
    ATTACHMENT_RULE_PROMPT,
    MULTI_LABEL_PROMPT,
    REPAIR_PROMPT,
    SINGLE_LABEL_PROMPT,
    SYSTEM_PROMPT,
    build_repair_prompt,
    build_response_schema,
    build_system_prompt,
    build_user_prompt,
    user_prompt_names,
)
from ..result import UndecidedReason

logger = logging.getLogger(__name__)


class ClassifyStep(Step):
    """Calls the LLM and turns its reply into a decision, or an abstention.

    Does nothing when an earlier step (rules) already decided -- that is what lets a
    deterministic rule answer without a model call.

    A reply that is not usable JSON, or that names labels outside the label set, is fed
    back to the model for up to ``max_repair_attempts`` further calls. A usable reply is
    then held to ``min_confidence``: single-label mode takes the top label if it reaches
    the threshold and is not tied; multi-label mode takes every label that reaches it.
    Otherwise the result is ``UNDECIDED`` -- never the nearest label.

    When a provider has a native structured-output mode and ``native_structured_output``
    is on (the default), each call also passes it the reply's JSON Schema (label names
    as an enum); otherwise the prompt alone describes the reply. Which path produced the
    result is ``metadata["structured_output"]``; the prompt refs of the deciding
    provider's last call are ``metadata["prompts"]``.

    ``fallbacks`` are further providers tried in order, each with its own repair budget,
    whenever the one before did not reach a decision: it was unreachable, its output
    was unusable, or it abstained. A cheap model answers the clear cases; a stronger one
    sees only the rest. The provider whose answer was used is ``metadata["llm"]``.
    Escalation follows :func:`~kuhaku.tools.classification.escalation.escalate`:
    :class:`~kuhaku.core.llm.base.LLMUnavailableError` is raised only when no provider
    produced any reply at all -- an earlier provider's abstention is still an answer
    about the text, a dropped connection is not. With attachments, providers that
    cannot take them are skipped up front.
    """

    name = CLASSIFY_STEP

    def __init__(
        self,
        *,
        instructions: str = "",
        max_repair_attempts: int = 1,
        fallbacks: Sequence[LLMProvider] = (),
        native_structured_output: bool = True,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")
        self.instructions = instructions
        self.max_repair_attempts = max_repair_attempts
        self.fallbacks = list(fallbacks)
        self.native_structured_output = native_structured_output

    def prepare(self, ctx: ClassificationContext) -> None:
        if ctx.attachments:
            ctx.instructions.append(ctx.prompts.get(ATTACHMENT_RULE_PROMPT).render())

    def run(self, ctx: ClassificationContext) -> None:
        if ctx.settled:
            return

        system = build_system_prompt(
            multi_label=ctx.multi_label,
            instructions=self.instructions,
            extra_rules=ctx.instructions,
            prompts=ctx.prompts,
        )
        user = build_user_prompt(
            labels=ctx.labels,
            text=ctx.text,
            history=ctx.history,
            blocks=ctx.prompt_blocks,
            attachments=ctx.attachments,
            prompts=ctx.prompts,
        )
        names = user_prompt_names(ctx.text, ctx.attachments)
        prompts = _Prompts(
            system=system,
            user=user,
            refs={
                "system": ctx.prompts.get(SYSTEM_PROMPT).ref,
                "mode": ctx.prompts.get(
                    MULTI_LABEL_PROMPT if ctx.multi_label else SINGLE_LABEL_PROMPT
                ).ref,
                "user": ctx.prompts.get(names["user"]).ref,
                "attachments": (
                    ctx.prompts.get(names["attachments"]).ref
                    if "attachments" in names
                    else None
                ),
            },
            repair=ctx.prompts.get(REPAIR_PROMPT),
        )

        providers = capable_providers(
            [ctx.llm, *self.fallbacks],
            {part.type for part in ctx.attachments},
            what="classification",
        )
        chosen = escalate(
            providers,
            lambda llm: self._attempt(ctx, llm, prompts),
            log=logger,
            what="classification",
        )
        outcome = chosen.value
        ctx.raw_output = outcome.raw_output
        ctx.metadata["llm"] = chosen.llm.name
        ctx.metadata["structured_output"] = "native" if outcome.native else "prompt"
        ctx.metadata["prompts"] = outcome.prompts
        if isinstance(outcome.answer, OutputParseError):
            ctx.metadata.pop("rationale", None)
            ctx.abstain(outcome.answer.reason, by=self.name, reason=str(outcome.answer))
        else:
            self._apply(ctx, outcome.answer)

    def _attempt(
        self, ctx: ClassificationContext, llm: LLMProvider, prompts: _Prompts
    ) -> Attempt[_Outcome]:
        """One provider's attempt plus its repairs. Accepted when its reply decides."""

        native = uses_native(llm, self.native_structured_output)
        schema = build_response_schema(ctx.labels)
        prompt = prompts.user
        refs = dict(prompts.refs)
        error: OutputParseError | None = None
        raw: str | None = None
        replied = False
        for _ in range(1 + self.max_repair_attempts):
            try:
                with start_span("classification.generate", llm=llm.name):
                    record_prompts(**refs)
                    completion = ctx.complete(
                        llm,
                        prompts.system,
                        prompt,
                        json_schema=schema,
                        native=self.native_structured_output,
                    )
            except LLMError:
                if not replied:
                    raise
                break  # what it already said is still an answer about the text
            replied = True
            raw = completion.text
            try:
                reply = parse_reply(raw, ctx.labels)
            except OutputParseError as exc:
                error = exc
                prompt = build_repair_prompt(
                    prompts.user, raw, exc.problems, prompts.repair
                )
                refs["user"] = prompts.repair.ref
                continue
            outcome = _Outcome(reply, raw, native, prompt_metadata(**refs))
            decided = _decision(ctx, reply)[0] == "decide"
            return Attempt(outcome, accepted=decided, rank=1)
        assert error is not None
        return Attempt(_Outcome(error, raw, native, prompt_metadata(**refs)), False, 0)

    def _apply(self, ctx: ClassificationContext, reply: Reply) -> bool:
        """Record ``reply`` on ``ctx`` as a decision or an abstention; True if decided."""

        ctx.scores = dict(reply.scores)
        if reply.rationale:
            ctx.metadata["rationale"] = reply.rationale
        else:
            ctx.metadata.pop("rationale", None)
        kind, detail, reason = _decision(ctx, reply)
        if kind == "decide":
            ctx.decide(detail, by=self.name, reason=reason)
            return True
        ctx.abstain(detail, by=self.name, reason=reason)
        return False


def _decision(ctx: ClassificationContext, reply: Reply) -> tuple[str, Any, str]:
    """What ``reply`` means under ``ctx``'s thresholds, without recording it:
    ``("decide", labels, reason)`` or ``("abstain", UndecidedReason, reason)``."""

    if not reply.scores:
        return (
            "abstain",
            UndecidedReason.NO_LABEL,
            (reply.rationale or "the model found no fitting label"),
        )
    ranked = sorted(reply.scores.items(), key=lambda item: -item[1])
    passing = [name for name, score in ranked if score >= ctx.min_confidence]
    if not passing:
        top, score = ranked[0]
        return (
            "abstain",
            UndecidedReason.LOW_CONFIDENCE,
            (
                f"best label '{top}' scored {score:g}, below min_confidence {ctx.min_confidence:g}"
            ),
        )
    if not ctx.multi_label:
        tied = [name for name, score in ranked if score == ranked[0][1]]
        if len(tied) > 1:
            return (
                "abstain",
                UndecidedReason.AMBIGUOUS,
                (f"labels {tied} tied at confidence {ranked[0][1]:g}"),
            )
        passing = passing[:1]
    return "decide", passing, reply.rationale or f"the model chose {passing}"


@dataclass(frozen=True)
class _Outcome:
    answer: Reply | OutputParseError
    raw_output: str | None
    native: bool
    prompts: dict[str, Any]


@dataclass(frozen=True)
class _Prompts:
    system: str
    user: str
    refs: dict[str, PromptRef | None]
    repair: PromptTemplate
