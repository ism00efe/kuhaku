"""What to extract: a Pydantic model plus the name, description and instructions that
travel with it.

The model is the single source of truth -- the JSON Schema shown to the LLM is derived
from it, never written by hand next to it, so the two cannot drift apart.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class ExtractionSchema(Generic[M]):
    """A Pydantic model registered for extraction.

    ``name`` defaults to the model's class name. ``description`` defaults to the model's
    own docstring -- read from the class itself, not inherited, so a model without one
    does not pick up ``BaseModel``'s. ``instructions`` is the caller-owned part of the
    system prompt: domain rules for this schema (e.g. "copy product names exactly as
    written"), added after the framework's own rules, never replacing them.
    """

    model: type[M]
    name: str = ""
    description: str = ""
    instructions: str = ""

    def __post_init__(self) -> None:
        if not (isinstance(self.model, type) and issubclass(self.model, BaseModel)):
            raise TypeError(
                f"ExtractionSchema.model must be a pydantic BaseModel subclass, "
                f"got {self.model!r}."
            )
        if not self.name:
            object.__setattr__(self, "name", self.model.__name__)
        if not self.description:
            own_doc = self.model.__dict__.get("__doc__")
            if own_doc:
                object.__setattr__(self, "description", inspect.cleandoc(own_doc))

    def json_schema(self) -> dict[str, Any]:
        return self.model.model_json_schema()


def as_schema(schema: ExtractionSchema[M] | type[M]) -> ExtractionSchema[M]:
    """Accept either a bare model class or an already-built :class:`ExtractionSchema`."""

    if isinstance(schema, ExtractionSchema):
        return schema
    return ExtractionSchema(model=schema)
