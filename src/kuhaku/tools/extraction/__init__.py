"""Structured extraction: free text -> a validated Pydantic model, as a second kuhaku tool.

Built only on ``kuhaku.core`` (the LLM abstraction and shared models) and pydantic; it
imports nothing from ``kuhaku.tools.rag``.

    from kuhaku import Extractor

    result = Extractor({"product": str, "quantity": float}).extract("5 boxes of tea please")
    result.status, result.data, result.missing_slots

A Pydantic model class works in place of the field description; see
:mod:`.schema` for every form a schema can take.
"""

from .config import ExtractionSettings
from .context import ExtractionContext
from .extractor import Extractor
from .parsing import OutputParseError
from .pipeline import EXTRACT_STEP, Pipeline, Step
from .prompts import FIXED_RULES, ExtractionPrompts
from .result import (
    ExtractionResult,
    ExtractionStatus,
    FieldReport,
    Issue,
    IssueKind,
    MissingField,
    Provenance,
)
from .router import ExtractionRouter, InputBlocked, LLMRoute, Route, RouteRequest
from .schema import ExtractionSchema
from .steps import (
    SECURITY_STEP,
    SLOT_FILLING_STEP,
    EvidenceStep,
    ExtractStep,
    SecurityStep,
    SlotFillingStep,
    ValidationStep,
    Validator,
)

__all__ = [
    "EXTRACT_STEP",
    "FIXED_RULES",
    "SECURITY_STEP",
    "SLOT_FILLING_STEP",
    "EvidenceStep",
    "ExtractStep",
    "ExtractionContext",
    "ExtractionPrompts",
    "ExtractionResult",
    "ExtractionRouter",
    "ExtractionSchema",
    "ExtractionSettings",
    "ExtractionStatus",
    "Extractor",
    "FieldReport",
    "InputBlocked",
    "Issue",
    "IssueKind",
    "LLMRoute",
    "MissingField",
    "OutputParseError",
    "Pipeline",
    "Provenance",
    "Route",
    "RouteRequest",
    "SecurityStep",
    "SlotFillingStep",
    "Step",
    "ValidationStep",
    "Validator",
]
