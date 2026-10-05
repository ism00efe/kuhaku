"""Structured extraction: free text -> a validated Pydantic model, as a second kuhaku tool.

Built only on ``kuhaku.core`` (the LLM abstraction and shared models) and pydantic; it
imports nothing from ``kuhaku.tools.rag``.

    from pydantic import BaseModel
    from kuhaku.tools.extraction import Extractor, EvidenceStep

    class Order(BaseModel):
        product: str
        quantity: float

    extractor = Extractor(Order, steps=[EvidenceStep()])
    result = extractor.extract("5 boxes of tea please")
    result.status, result.value, result.missing, result.fields
"""

from .context import ExtractionContext
from .extractor import Extractor
from .parsing import OutputParseError
from .pipeline import EXTRACT_STEP, Pipeline, Step
from .result import (
    ExtractionResult,
    ExtractionStatus,
    FieldReport,
    Issue,
    IssueKind,
    Provenance,
)
from .router import ExtractionRouter, LLMRoute, Route, RouteRequest
from .schema import ExtractionSchema
from .steps import (
    CONTINUATION_STEP,
    ContinuationStep,
    EvidenceStep,
    ExtractStep,
    ValidationStep,
    Validator,
)

__all__ = [
    "CONTINUATION_STEP",
    "EXTRACT_STEP",
    "ContinuationStep",
    "EvidenceStep",
    "ExtractStep",
    "ExtractionContext",
    "ExtractionResult",
    "ExtractionRouter",
    "ExtractionSchema",
    "ExtractionStatus",
    "Extractor",
    "FieldReport",
    "Issue",
    "IssueKind",
    "LLMRoute",
    "OutputParseError",
    "Pipeline",
    "Provenance",
    "Route",
    "RouteRequest",
    "Step",
    "ValidationStep",
    "Validator",
]
