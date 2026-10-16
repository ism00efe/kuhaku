"""What to extract: a Pydantic model plus the name, description and instructions that
travel with it.

The model is the single source of truth -- the JSON Schema shown to the LLM is derived
from it, never written by hand next to it, so the two cannot drift apart.

A caller who does not want to write a model class can describe the fields instead and
the model is built for them:

    ["name", "email"]                                   # every field a required str
    {"name": str, "age": int, "note": str | None}       # a type per field; None = optional
    {"total": (float, "grand total incl. VAT")}         # (type, description)
    {"lines": [{"item": str, "price": float}]}          # nested objects and lists of them
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, create_model

M = TypeVar("M", bound=BaseModel)

FieldSpec = Mapping[str, Any] | Sequence[str]
"""A field description in place of a model class -- see the module docstring."""

_DEFAULT_NAME = "Extraction"


@dataclass(frozen=True)
class ExtractionSchema(Generic[M]):
    """A Pydantic model (or a field description) registered for extraction.

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
        if _is_field_spec(self.model):
            built = model_from_fields(self.model, self.name or _DEFAULT_NAME)  # type: ignore[arg-type]
            object.__setattr__(self, "model", built)
        if not (isinstance(self.model, type) and issubclass(self.model, BaseModel)):
            raise TypeError(
                "ExtractionSchema.model must be a pydantic BaseModel subclass, a dict of "
                f"field name -> type, or a list of field names; got {self.model!r}."
            )
        if not self.name:
            object.__setattr__(self, "name", self.model.__name__)
        if not self.description:
            own_doc = self.model.__dict__.get("__doc__")
            if own_doc:
                object.__setattr__(self, "description", inspect.cleandoc(own_doc))

    def json_schema(self) -> dict[str, Any]:
        return self.model.model_json_schema()


def as_schema(
    schema: ExtractionSchema[M] | type[M] | FieldSpec,
    *,
    name: str = "",
    description: str = "",
    instructions: str = "",
) -> ExtractionSchema[M]:
    """Accept a model class, a field description or an already-built schema."""

    if isinstance(schema, ExtractionSchema):
        if name or description or instructions:
            raise TypeError(
                "name/description/instructions belong on the ExtractionSchema already "
                "passed; set them there."
            )
        return schema
    return ExtractionSchema(
        model=schema,  # type: ignore[arg-type]
        name=name,
        description=description,
        instructions=instructions,
    )


def model_from_fields(fields: FieldSpec, name: str) -> type[BaseModel]:
    """Build a model class from a field description (see the module docstring)."""

    if isinstance(fields, Mapping):
        items = list(fields.items())
    else:
        items = [(field_name, str) for field_name in fields]
    if not items:
        raise TypeError("A field description needs at least one field.")

    definitions: dict[str, Any] = {}
    for field_name, spec in items:
        if not isinstance(field_name, str) or not field_name:
            raise TypeError(f"Field names must be non-empty strings; got {field_name!r}.")
        description = None
        if isinstance(spec, tuple):
            if len(spec) != 2 or not isinstance(spec[1], str):
                raise TypeError(
                    f"Field '{field_name}': a tuple must be (type, description); got {spec!r}."
                )
            spec, description = spec
        annotation = _annotation(spec, f"{name}_{field_name}")
        default = None if _accepts_none(annotation) else ...
        definitions[field_name] = (annotation, Field(default, description=description))
    return create_model(_class_name(name), **definitions)  # type: ignore[call-overload,no-any-return]


def _annotation(spec: Any, nested_name: str) -> Any:
    if isinstance(spec, Mapping):
        return model_from_fields(spec, nested_name)
    if isinstance(spec, list):
        if len(spec) != 1:
            raise TypeError(
                f"A list field is written as [item type]; got {spec!r} for '{nested_name}'."
            )
        return list[_annotation(spec[0], nested_name)]  # type: ignore[misc]
    return spec


def _accepts_none(annotation: Any) -> bool:
    if annotation is None or annotation is type(None) or annotation is Any:
        return True
    return type(None) in typing.get_args(annotation) and (
        typing.get_origin(annotation) in (typing.Union, types.UnionType)
    )


def _is_field_spec(value: Any) -> bool:
    if isinstance(value, Mapping):
        return True
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _class_name(name: str) -> str:
    # create_model needs an identifier; the schema keeps the caller's name unchanged.
    cleaned = re.sub(r"\W", "_", name)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"_{cleaned}"
