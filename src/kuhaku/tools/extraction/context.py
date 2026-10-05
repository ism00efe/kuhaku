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

from .result import (
    ExtractionResult,
    ExtractionStatus,
    FieldReport,
    Issue,
    status_from_issues,
)
from .schema import ExtractionSchema


@dataclass
class ExtractionContext:
    schema: ExtractionSchema[Any]
    text: str
    llm: LLMProvider
    history: Sequence[Message] = ()
    reference: Any = None
    prior_data: dict[str, Any] | None = None

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

    def to_result(self) -> ExtractionResult[Any]:
        status = status_from_issues(self.issues)
        return ExtractionResult(
            status=status,
            schema_name=self.schema.name,
            value=self.value if status is ExtractionStatus.COMPLETE else None,
            data=self.data,
            issues=tuple(self.issues),
            fields=dict(self.fields),
            attempts=self.attempts,
            metadata=dict(self.metadata),
        )
