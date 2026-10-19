"""Wire schemas shared by every transport.

Pydantic only (a core dependency), so these import without either optional extra. The
HTTP API uses them as request/response models; the MCP server derives its tools'
``inputSchema``/``outputSchema`` from the same classes, so the two transports cannot
drift apart.

Request models forbid unknown fields: identity and roles come from the authenticator,
and a body that tries to supply them is rejected rather than ignored.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from kuhaku.core.sanitization import Redaction
from kuhaku.tools.extraction import ExtractionResult, ExtractionSchema, ExtractionStatus
from kuhaku.tools.rag.models import Answer

from .limits import (
    MAX_CONTEXT_TEXT_CHARS,
    MAX_EXTRACT_TEXT_CHARS,
    MAX_INGEST_TEXT_CHARS,
    MAX_QUESTION_CHARS,
)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(_Request):
    question: str = Field(
        min_length=1, max_length=MAX_QUESTION_CHARS, description="The question to answer."
    )
    context_text: str | None = Field(
        default=None,
        max_length=MAX_CONTEXT_TEXT_CHARS,
        description="Optional supplementary context (JSON, XML or text) folded into "
        "the retrieval query.",
    )


class CitationOut(BaseModel):
    tag: str
    document_id: str
    title: str
    doc_type: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    abstained: bool = Field(
        description="True when retrieval found nothing the caller may see, so no answer "
        "was generated."
    )
    redactions: list[str] = Field(description="Categories masked in the question.")
    trace_id: str

    @classmethod
    def from_answer(cls, answer: Answer) -> AskResponse:
        # source_path is left out on purpose: for a corpus loaded from disk it is a
        # server-side path, which is nothing a remote caller needs.
        return cls(
            answer=answer.text,
            citations=[
                CitationOut(
                    tag=c.tag,
                    document_id=c.document_id,
                    title=c.title,
                    doc_type=c.doc_type,
                    score=c.score,
                )
                for c in answer.citations
            ],
            abstained=answer.abstained,
            redactions=list(answer.redactions),
            trace_id=answer.trace_id,
        )


class IngestRequest(_Request):
    text: str = Field(
        min_length=1, max_length=MAX_INGEST_TEXT_CHARS, description="The document's text."
    )
    filename: str = Field(
        default="document.txt",
        min_length=1,
        max_length=255,
        description="Document name; becomes its title and citation source.",
    )
    access_tags: list[str] | None = Field(
        default=None,
        max_length=64,
        description="Restrict the document to callers holding at least one of these "
        "roles; each must be one of the caller's own roles. Omitted or empty: visible "
        "to every caller, which the server's policy must allow separately "
        "('ingest_untagged').",
    )


class RedactionOut(BaseModel):
    label: str
    count: int


class IngestResponse(BaseModel):
    chunks_added: int
    redactions: list[RedactionOut]
    trace_id: str

    @classmethod
    def build(cls, chunks_added: int, redactions: list[Redaction], trace_id: str) -> IngestResponse:
        return cls(
            chunks_added=chunks_added,
            redactions=[RedactionOut(label=r.label, count=r.count) for r in redactions],
            trace_id=trace_id,
        )


class ExtractRequest(_Request):
    text: str = Field(max_length=MAX_EXTRACT_TEXT_CHARS, description="The text to extract from.")
    reference: Any = Field(
        default=None,
        description="Optional data (e.g. the current cart) shown to the model only to "
        "resolve what the text refers to.",
    )
    prior: dict[str, Any] | None = Field(
        default=None,
        description="The data of an earlier extraction, to continue it.",
    )


class IssueOut(BaseModel):
    kind: str
    code: str
    message: str
    path: str | None = None


class FieldReportOut(BaseModel):
    provenance: str
    evidence: str | None = None


class ExtractionResponse(BaseModel):
    """The result of one extraction. ``value`` is typed per extractor -- see
    :func:`extraction_response_model`."""

    status: ExtractionStatus
    schema_name: str
    value: Any = None
    data: dict[str, Any] | None = None
    missing: list[str]
    question: str | None = None
    issues: list[IssueOut]
    fields: dict[str, FieldReportOut]
    attempts: int
    trace_id: str


def extraction_response_model(schema: ExtractionSchema[Any]) -> type[ExtractionResponse]:
    """An :class:`ExtractionResponse` subclass whose ``value`` is ``schema``'s model, so
    the OpenAPI document and the MCP ``outputSchema`` both carry the real shape."""

    model: type[ExtractionResponse] = create_model(  # type: ignore[call-overload]
        f"{schema.name}ExtractionResponse",
        __base__=ExtractionResponse,
        value=(schema.model | None, None),
    )
    return model


def extraction_payload(result: ExtractionResult[Any]) -> dict[str, Any]:
    """``result`` as the JSON-ready dict both transports send.

    ``metadata`` (model names, structured-output mode) stays server-side.
    """

    value = result.value.model_dump(mode="json", by_alias=True) if result.value is not None else None
    return {
        "status": result.status.value,
        "schema_name": result.schema_name,
        "value": value,
        "data": result.data,
        "missing": list(result.missing_slots),
        "question": result.question,
        "issues": [
            {"kind": i.kind.value, "code": i.code, "message": i.message, "path": i.path}
            for i in result.issues
        ],
        "fields": {
            path: {"provenance": f.provenance.value, "evidence": f.evidence}
            for path, f in result.fields.items()
        },
        "attempts": result.attempts,
        "trace_id": result.trace_id,
    }


def json_schema(model: type[BaseModel], *, output: bool = False) -> dict[str, Any]:
    """``model``'s JSON Schema, self-contained (``$defs`` at the root) and copied so a
    caller mutating it cannot reach pydantic's cache. ``output=True`` describes what
    serializing produces (aliases, computed defaults) rather than what validation
    accepts."""

    mode: Literal["validation", "serialization"] = "serialization" if output else "validation"
    return copy.deepcopy(model.model_json_schema(mode=mode))
