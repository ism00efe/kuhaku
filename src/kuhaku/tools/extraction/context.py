"""The state one extraction carries through the pipeline.

Steps communicate only through this object -- a step never calls another step -- which
is what lets any of them be added, removed or swapped without touching the rest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from kuhaku.core.llm.base import LLMProvider
from kuhaku.core.models import Message

from .prompts import ExtractionPrompts
from .result import (
    ExtractionResult,
    ExtractionStatus,
    FieldReport,
    Issue,
    IssueKind,
    MissingField,
    clarification_question,
    status_from_issues,
)
from .schema import ExtractionSchema, describe_path


@dataclass
class ExtractionContext:
    schema: ExtractionSchema[Any]
    text: str
    llm: LLMProvider
    history: Sequence[Message] = ()
    reference: Any = None
    prior_data: dict[str, Any] | None = None
    trace_id: str = ""
    prompts: ExtractionPrompts = field(default_factory=ExtractionPrompts)

    # Written during the prepare phase: steps that need something from the model (e.g.
    # evidence) add a system-prompt rule, a top-level response key or a titled block of
    # input here, so the extract step never has to know which other steps exist.
    instructions: list[str] = field(default_factory=list)
    response_fields: dict[str, str] = field(default_factory=dict)
    prompt_blocks: list[tuple[str, str]] = field(default_factory=list)

    # Written during the run phase.
    raw_output: str | None = None
    response: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    value: BaseModel | None = None
    issues: list[Issue] = field(default_factory=list)
    fields: dict[str, FieldReport] = field(default_factory=dict)
    attempts: int = 0
    stopped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def stop(self) -> None:
        """Skip every remaining step -- for when continuing would only operate on
        output that does not exist."""

        self.stopped = True

    def to_result(self, question_template: str = "") -> ExtractionResult[Any]:
        """The result of this extraction. ``question_template`` (its ``{fields}``
        placeholder filled in) becomes ``result.question`` when values are missing."""

        status = status_from_issues(self.issues)
        missing = tuple(
            MissingField(issue.path, *describe_path(self.schema.model, issue.path))
            for issue in self.issues
            if issue.kind is IssueKind.NEEDS_CLARIFICATION and issue.path is not None
        )
        question = (
            clarification_question(question_template, missing)
            if status is ExtractionStatus.NEEDS_CLARIFICATION and question_template
            else None
        )
        return ExtractionResult(
            status=status,
            schema_name=self.schema.name,
            value=self.value if status is ExtractionStatus.COMPLETE else None,
            data=self.data,
            issues=tuple(self.issues),
            fields=dict(self.fields),
            attempts=self.attempts,
            metadata=dict(self.metadata),
            trace_id=self.trace_id,
            missing_fields=missing,
            question=question,
        )
