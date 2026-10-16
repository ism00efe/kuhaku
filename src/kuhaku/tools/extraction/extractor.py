"""``Extractor``: the entry point -- one schema, one pipeline, one LLM provider."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from kuhaku.core.config import Settings, get_settings
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.llm import build_llm_provider, with_runtime
from kuhaku.core.llm.base import LLMProvider
from kuhaku.core.models import Message
from kuhaku.core.observability import traced_call
from kuhaku.core.sanitization import sanitize
from kuhaku.core.security import record_audit
from kuhaku.evaluation.sample import EvaluationSample

from .config import ExtractionSettings
from .context import ExtractionContext
from .pipeline import EXTRACT_STEP, Pipeline, Step
from .prompts import ExtractionPrompts, as_prompts
from .result import ExtractionResult
from .schema import ExtractionSchema, FieldSpec, as_schema
from .steps.extract import ExtractStep
from .steps.security import SecurityStep
from .steps.slot_filling import SLOT_FILLING_STEP, SlotFillingStep

M = TypeVar("M", bound=BaseModel)

logger = logging.getLogger(__name__)


class Extractor(Generic[M]):
    """Extracts one Pydantic model's worth of data from free text.

    ``schema`` is a Pydantic model class, or -- to try something without writing a
    class -- a field description such as ``{"name": str, "age": int}`` or
    ``["name", "email"]`` (see :mod:`kuhaku.tools.extraction.schema`). ``name``,
    ``description`` and ``instructions`` are optional and travel with the schema.

    ``prompts`` replaces any of the tool's default prompt texts -- an
    :class:`~kuhaku.tools.extraction.prompts.ExtractionPrompts` or a dict such as
    ``{"rules": "..."}``; :meth:`preview` shows the exact prompt for a text.

    The default pipeline is ``security`` (which does nothing with every check off), then
    ``extract``, then ``slot_filling`` (which does nothing unless ``prior`` is passed). Optional steps are passed as ``steps`` (appended after
    those) or composed afterwards through :attr:`pipeline` --
    ``add(..., before=/after=)``, ``remove``, ``replace``, ``enable``/``disable``.

    ``llm`` takes any :class:`~kuhaku.core.llm.base.LLMProvider`; when omitted, the
    provider is built from ``settings`` (or the environment) through
    :func:`~kuhaku.core.llm.build_llm_provider`, the same path ``RAG`` uses.
    ``fallback_llms`` are tried in order when ``llm`` cannot produce usable output --
    see :class:`~kuhaku.tools.extraction.steps.ExtractStep`.

    Every provider goes through :func:`~kuhaku.core.llm.with_runtime`, as ``RAG``'s
    does, so whatever core applies to model calls applies here too;
    ``enable_token_tracking=False`` leaves out token accounting.

    ``audit_log_path`` turns on core's audit log: one JSON line per :meth:`extract` call,
    whatever the outcome, holding a hash of the text and a PII-masked excerpt, never the
    raw text. Off unless a path is given -- it writes a file into the caller's project.

    ``max_retries`` and ``audit_log_path`` left as ``None`` come from
    ``extraction_settings``, which defaults to :class:`ExtractionSettings` read from
    ``KUHAKU_EXTRACTION__*`` environment variables -- the settings every extraction
    shares, the router's schema choice included. Its ``security`` options are run by a
    :class:`SecurityStep` at the front of every pipeline; a step named ``security``
    passed in ``steps`` is used instead.
    """

    def __init__(
        self,
        schema: ExtractionSchema[M] | type[M] | FieldSpec,
        *,
        name: str = "",
        description: str = "",
        instructions: str = "",
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
        steps: Sequence[Step] = (),
        max_retries: int | None = None,
        fallback_llms: Sequence[LLMProvider] = (),
        prompts: ExtractionPrompts | Mapping[str, str] | None = None,
        enable_token_tracking: bool = True,
        audit_log_path: str | None = None,
        extraction_settings: ExtractionSettings | None = None,
    ) -> None:
        es = extraction_settings or ExtractionSettings()
        self.extraction_settings = es
        if max_retries is None:
            max_retries = es.max_retries
        self.audit_log_path = audit_log_path if audit_log_path is not None else es.audit_log_path
        self.prompts: ExtractionPrompts = as_prompts(prompts)
        self.schema: ExtractionSchema[M] = as_schema(
            schema, name=name, description=description, instructions=instructions
        )
        built = llm if llm is not None else build_llm_provider(settings or get_settings())
        self.llm: LLMProvider = with_runtime(built, token_tracking=enable_token_tracking)
        self.fallback_llms: list[LLMProvider] = [
            with_runtime(fallback, token_tracking=enable_token_tracking)
            for fallback in fallback_llms
        ]
        given = {step.name for step in steps}
        security = [] if SecurityStep.name in given else [SecurityStep(es.security)]
        self.pipeline = Pipeline(
            [
                *security,
                ExtractStep(
                    max_retries=max_retries, fallbacks=self.fallback_llms
                ),
                SlotFillingStep(),
                *steps,
            ]
        )

    def extract(
        self,
        text: str,
        *,
        history: Sequence[Message] = (),
        reference: Any = None,
        prior: ExtractionResult[Any] | dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> ExtractionResult[M]:
        """Extract from ``text``.

        ``history`` (earlier conversation turns) and ``reference`` (any JSON-serializable
        data, or a string -- e.g. the current cart) are shown to the model only for
        resolving what ``text`` refers to; values are extracted from ``text``.

        ``prior`` continues an earlier extraction -- the previous result, or its
        ``data`` dict -- see :class:`SlotFillingStep`.

        ``trace_id`` binds every log record and span of this call to an id; without
        one, an id bound by the caller is inherited, else a new one is minted.
        """

        with traced_call("extraction", logger, trace_id=trace_id, schema=self.schema.name) as (
            tid,
            rec,
        ):
            ctx = self._context(text, history, reference, prior, tid)
            try:
                self.pipeline.run(ctx)
            except Exception as exc:
                self._audit(text, tid, status="error", error=type(exc).__name__)
                raise
            result = ctx.to_result(self.extraction_settings.clarification_question)
            rec.set(outcome=result.status.value, attempts=result.attempts)
            self._audit(
                text,
                tid,
                status=result.status.value,
                issues=[issue.code for issue in result.issues],
                llm=result.metadata.get("llm"),
            )
            return result

    def _audit(self, text: str, trace_id: str, **details: Any) -> None:
        if self.audit_log_path is None:
            return
        record_audit(
            self.audit_log_path,
            trace_id=trace_id,
            raw_question=text,
            sanitized_retrieval_query=sanitize(text),
            event_type="extraction",
            metadata={"schema": self.schema.name, **details},
        )


    def evaluate_sample(self, query: str) -> EvaluationSample:
        """Implements kuhaku's ``EvaluationTarget``: extracts from ``query`` and answers
        with the extracted data as a JSON object, for
        :class:`~kuhaku.evaluation.StructuredMatchMetric` to score against a
        ``golden_answer`` holding the expected object as JSON. Status, schema and missing
        fields ride along in ``metadata``."""

        return evaluation_sample(query, self.extract(query), answer_data=None)

    def preview(
        self,
        text: str,
        *,
        history: Sequence[Message] = (),
        reference: Any = None,
        prior: ExtractionResult[Any] | dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """The ``(system, user)`` prompt the first model call for ``text`` would send,
        with every enabled step's contribution -- without calling the model. The
        boundary markers are random per call, so they differ from a real run's.

        A replaced ``extract`` step is previewed through its own ``build_prompts(ctx)``;
        one without it raises :class:`ConfigError`."""

        ctx = self._context(text, history, reference, prior, "")
        self.pipeline.prepare(ctx)
        step = self.pipeline.get(EXTRACT_STEP)
        builder = getattr(step, "build_prompts", None)
        if not callable(builder):
            # Showing ExtractStep's prompt here would present text the replacement may
            # never send.
            raise ConfigError(
                f"The '{EXTRACT_STEP}' step was replaced by {type(step).__name__}, which "
                f"has no build_prompts(ctx), so there is no prompt to preview. Give it a "
                f"build_prompts(ctx) -> (system, user) method."
            )
        system, user = builder(ctx)
        return system, user

    def _context(
        self,
        text: str,
        history: Sequence[Message],
        reference: Any,
        prior: ExtractionResult[Any] | dict[str, Any] | None,
        trace_id: str,
    ) -> ExtractionContext:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}.")
        if not all(isinstance(message, Message) for message in history):
            raise TypeError("history must be a sequence of kuhaku Message objects.")
        prior_data = prior.data if isinstance(prior, ExtractionResult) else prior
        if prior_data is not None and not isinstance(prior_data, dict):
            raise TypeError(
                f"prior must be an ExtractionResult or a dict, got {type(prior).__name__}."
            )
        if prior_data and not (
            SLOT_FILLING_STEP in self.pipeline and self.pipeline.is_enabled(SLOT_FILLING_STEP)
        ):
            raise ConfigError(
                f"prior= was given but the '{SLOT_FILLING_STEP}' step is not enabled, so it "
                f"would be silently ignored. Enable it or stop passing prior=."
            )
        return ExtractionContext(
            schema=self.schema,
            text=text,
            llm=self.llm,
            history=tuple(history),
            reference=reference,
            prior_data=prior_data,
            trace_id=trace_id,
            prompts=self.prompts,
        )

def evaluation_sample(
    query: str, result: ExtractionResult[Any], *, answer_data: dict[str, Any] | None
) -> EvaluationSample:
    data = answer_data if answer_data is not None else result.data
    return EvaluationSample(
        query=query,
        answer=None if data is None else json.dumps(data, ensure_ascii=False, default=str),
        metadata={
            "status": result.status.value,
            "schema": result.schema_name,
            "missing_slots": list(result.missing_slots),
            "trace_id": result.trace_id,
        },
    )

