"""``Extractor``: the entry point -- one schema, one pipeline, one LLM provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from kuhaku.core.config import Settings, get_settings
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.base import LLMProvider
from kuhaku.core.models import Message

from .context import ExtractionContext
from .pipeline import Pipeline, Step
from .result import ExtractionResult
from .schema import ExtractionSchema, as_schema
from .steps.continuation import CONTINUATION_STEP, ContinuationStep
from .steps.extract import ExtractStep

M = TypeVar("M", bound=BaseModel)


class Extractor(Generic[M]):
    """Extracts one Pydantic model's worth of data from free text.

    The default pipeline is ``extract`` followed by ``continuation`` (which does nothing
    unless ``prior`` is passed). Optional steps are passed as ``steps`` (appended after
    those) or composed afterwards through :attr:`pipeline` --
    ``add(..., before=/after=)``, ``remove``, ``replace``, ``enable``/``disable``.

    ``llm`` takes any :class:`~kuhaku.core.llm.base.LLMProvider`; when omitted, the
    provider is built from ``settings`` (or the environment) through
    :func:`~kuhaku.core.llm.build_llm_provider`, the same path ``RAG`` uses.
    """

    def __init__(
        self,
        schema: ExtractionSchema[M] | type[M],
        *,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
        steps: Sequence[Step] = (),
        max_repair_attempts: int = 1,
    ) -> None:
        self.schema: ExtractionSchema[M] = as_schema(schema)
        self.llm: LLMProvider = (
            llm if llm is not None else build_llm_provider(settings or get_settings())
        )
        self.pipeline = Pipeline(
            [ExtractStep(max_repair_attempts=max_repair_attempts), ContinuationStep(), *steps]
        )

    def extract(
        self,
        text: str,
        *,
        history: Sequence[Message] = (),
        reference: Any = None,
        prior: ExtractionResult[Any] | dict[str, Any] | None = None,
    ) -> ExtractionResult[M]:
        """Extract from ``text``.

        ``history`` (earlier conversation turns) and ``reference`` (any JSON-serializable
        data, or a string -- e.g. the current cart) are shown to the model only for
        resolving what ``text`` refers to; values are extracted from ``text``.

        ``prior`` continues an earlier extraction -- the previous result, or its
        ``data`` dict -- see :class:`ContinuationStep`.
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
        ctx = ExtractionContext(
            schema=self.schema,
            text=text,
            llm=self.llm,
            history=tuple(history),
            reference=reference,
            prior_data=prior_data,
        )
        self.pipeline.run(ctx)
        return ctx.to_result()
