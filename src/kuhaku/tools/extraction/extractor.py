"""``Extractor``: the entry point -- one schema, one pipeline, one LLM provider."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from kuhaku.core.config import Settings, get_settings
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.base import LLMProvider
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.models import Message
from kuhaku.core.observability import bind_trace_id, instrumented_step

from .context import ExtractionContext
from .pipeline import Pipeline, Step
from .result import ExtractionResult
from .schema import ExtractionSchema, as_schema
from .steps.continuation import CONTINUATION_STEP, ContinuationStep
from .steps.extract import ExtractStep

M = TypeVar("M", bound=BaseModel)

logger = logging.getLogger(__name__)


class Extractor(Generic[M]):
    """Extracts one Pydantic model's worth of data from free text.

    The default pipeline is ``extract`` followed by ``continuation`` (which does nothing
    unless ``prior`` is passed). Optional steps are passed as ``steps`` (appended after
    those) or composed afterwards through :attr:`pipeline` --
    ``add(..., before=/after=)``, ``remove``, ``replace``, ``enable``/``disable``.

    ``llm`` takes any :class:`~kuhaku.core.llm.base.LLMProvider`; when omitted, the
    provider is built from ``settings`` (or the environment) through
    :func:`~kuhaku.core.llm.build_llm_provider`, the same path ``RAG`` uses.
    ``fallback_llms`` are tried in order when ``llm`` cannot produce usable output --
    see :class:`~kuhaku.tools.extraction.steps.ExtractStep`.

    Every provider is wrapped in :class:`~kuhaku.core.llm.token_tracking.TokenTrackingLLM`
    unless ``enable_token_tracking=False``, as ``RAG`` does.
    """

    def __init__(
        self,
        schema: ExtractionSchema[M] | type[M],
        *,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
        steps: Sequence[Step] = (),
        max_repair_attempts: int = 1,
        fallback_llms: Sequence[LLMProvider] = (),
        enable_token_tracking: bool = True,
    ) -> None:
        self.schema: ExtractionSchema[M] = as_schema(schema)
        built = llm if llm is not None else build_llm_provider(settings or get_settings())
        track = _token_tracked if enable_token_tracking else _untracked
        self.llm: LLMProvider = track(built)
        fallbacks = [track(fallback) for fallback in fallback_llms]
        self.pipeline = Pipeline(
            [
                ExtractStep(max_repair_attempts=max_repair_attempts, fallbacks=fallbacks),
                ContinuationStep(),
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
        ``data`` dict -- see :class:`ContinuationStep`.

        ``trace_id`` binds every log record and span of this call to an id; without
        one, an id bound by the caller is inherited, else a new one is minted.
        """

        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}.")
        prior_data = prior.data if isinstance(prior, ExtractionResult) else prior
        if prior_data and not (
            CONTINUATION_STEP in self.pipeline and self.pipeline.is_enabled(CONTINUATION_STEP)
        ):
            raise ConfigError(
                f"prior= was given but the '{CONTINUATION_STEP}' step is not enabled, so it "
                f"would be silently ignored. Enable it or stop passing prior=."
            )
        with (
            bind_trace_id(trace_id) as tid,
            instrumented_step("extraction", logger, schema=self.schema.name) as rec,
        ):
            ctx = ExtractionContext(
                schema=self.schema,
                text=text,
                llm=self.llm,
                history=tuple(history),
                reference=reference,
                prior_data=prior_data,
                trace_id=tid,
            )
            self.pipeline.run(ctx)
            result = ctx.to_result()
            rec.set(outcome=result.status.value, attempts=result.attempts)
            return result


def _token_tracked(llm: LLMProvider) -> LLMProvider:
    if isinstance(llm, TokenTrackingLLM):
        return llm
    # A provider's name is "<provider>:<model>"; the part before the colon is what
    # TokenTrackingLLM labels its metrics with.
    return TokenTrackingLLM(llm, provider=llm.name.split(":", 1)[0])


def _untracked(llm: LLMProvider) -> LLMProvider:
    return llm
