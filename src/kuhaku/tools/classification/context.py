"""The state one classification carries through the pipeline.

Steps communicate only through this object -- a step never calls another step -- which
is what lets any of them be added, removed or swapped without touching the rest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kuhaku.core.exceptions import CustomComponentError
from kuhaku.core.llm.base import LLMProvider, TokenUsage
from kuhaku.core.media import MediaPart
from kuhaku.core.models import Message
from kuhaku.core.prompts import PromptRegistry

from .escalation import Completion, complete_json
from .labels import LabelSet
from .prompts import DEFAULT_PROMPTS
from .result import (
    ClassificationResult,
    ClassificationStatus,
    UndecidedReason,
    add_usage,
)


@dataclass
class ClassificationContext:
    labels: LabelSet
    text: str
    llm: LLMProvider
    multi_label: bool = False
    min_confidence: float = 0.5
    history: Sequence[Message] = ()
    trace_id: str = ""
    # Images/documents that are part of the text to classify, alongside (or instead of)
    # ``text``. Rules match ``text`` only.
    attachments: tuple[MediaPart, ...] = ()
    # Where steps look their prompt templates up; the Classifier passes the caller's
    # registry layered over the built-in one.
    prompts: PromptRegistry = DEFAULT_PROMPTS

    # Written during the prepare phase: a step that needs something from the model adds
    # a system-prompt rule or a titled block of input here, so the classify step never
    # has to know which other steps exist.
    instructions: list[str] = field(default_factory=list)
    prompt_blocks: list[tuple[str, str]] = field(default_factory=list)

    # The outcome. ``status`` stays None until a step calls decide() or abstain(); the
    # classify step skips its model call when an earlier step (rules) already did.
    status: ClassificationStatus | None = None
    chosen: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    decided_by: str = ""
    reason: str = ""
    undecided: UndecidedReason | None = None

    raw_output: str | None = None
    response: dict[str, Any] | None = None
    attempts: int = 0
    usage: TokenUsage | None = None
    stopped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def settled(self) -> bool:
        """True once a step has decided or abstained."""

        return self.status is not None

    def decide(
        self,
        labels: Iterable[str],
        *,
        by: str,
        reason: str = "",
        scores: Mapping[str, float] | None = None,
    ) -> None:
        """Record a decision. ``labels`` must be names from the label set -- exactly one
        in single-label mode; in multi-label mode they are kept highest score first."""

        chosen = tuple(dict.fromkeys(labels))
        unknown = [name for name in chosen if name not in self.labels]
        if unknown:
            raise CustomComponentError(
                f"Step '{by}' decided on {unknown}, which are not in the label set "
                f"{list(self.labels.names)}."
            )
        if not chosen:
            raise CustomComponentError(
                f"Step '{by}' decided on no label; call abstain() to record no decision."
            )
        if not self.multi_label and len(chosen) != 1:
            raise CustomComponentError(
                f"Step '{by}' decided on {list(chosen)} in single-label mode."
            )
        if scores is not None:
            self.scores = dict(scores)
        self.chosen = tuple(sorted(chosen, key=lambda name: -self.scores.get(name, 0.0)))
        self.status = ClassificationStatus.DECIDED
        self.undecided = None
        self.decided_by = by
        self.reason = reason

    def abstain(self, why: UndecidedReason, *, by: str, reason: str = "") -> None:
        """Record that no label is chosen, and why. Overrides an earlier decision --
        that is how a later step vetoes one."""

        self.status = ClassificationStatus.UNDECIDED
        self.chosen = ()
        self.undecided = why
        self.decided_by = by
        self.reason = reason

    def generate(self, llm: LLMProvider, system: str, user: str) -> str:
        """Call ``llm`` with the prompt alone and account for the call (attempts, token
        usage). Steps that call a model go through here or :meth:`complete` so the
        result's totals include them."""

        return self.complete(llm, system, user, native=False).text

    def complete(
        self,
        llm: LLMProvider,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        name: str = "classification",
        native: bool = True,
    ) -> Completion:
        """One structured completion through
        :func:`~kuhaku.tools.classification.escalation.complete_json` -- the provider's native
        structured-output mode when it has one and ``json_schema`` is given --
        with this context's attachments, accounted for like :meth:`generate`. Usage is
        read off the response, not the provider's ``last_usage`` side channel."""

        self.attempts += 1
        completion = complete_json(
            llm,
            system,
            user,
            json_schema=json_schema,
            name=name,
            attachments=self.attachments,
            native=native,
        )
        self.usage = add_usage(self.usage, completion.usage)
        return completion

    def stop(self) -> None:
        """Skip every remaining step."""

        self.stopped = True

    def to_result(self) -> ClassificationResult:
        if self.status is None:
            raise CustomComponentError(
                "The pipeline finished without a step deciding or abstaining. The "
                "'classify' step always does one or the other; a replacement must too."
            )
        return ClassificationResult(
            status=self.status,
            labels=self.chosen,
            scores=dict(self.scores),
            decided_by=self.decided_by,
            reason=self.reason,
            undecided=self.undecided,
            attempts=self.attempts,
            usage=self.usage,
            metadata=dict(self.metadata),
            trace_id=self.trace_id,
        )
