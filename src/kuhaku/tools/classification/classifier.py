"""``Classifier``: the entry point -- one label set, one pipeline, one LLM provider."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from kuhaku.core.config import Settings, get_settings
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.base import LLMProvider
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.media import MediaPart
from kuhaku.core.models import Message
from kuhaku.core.observability import bind_trace_id, instrumented_step
from kuhaku.core.prompts import PromptRegistry
from kuhaku.core.session import ContextWindow, Session, loggable_session_id
from kuhaku.evaluation.sample import EvaluationSample

from .context import ClassificationContext
from .documents import Document, document_text, is_document
from .labels import Label, LabelSet, as_label_set
from .pipeline import Pipeline, Step
from .prompts import layered
from .result import ClassificationResult
from .steps.classify import ClassifyStep
from .steps.rules import Rule, RuleStep

logger = logging.getLogger(__name__)

SESSION_NAMESPACE = "classification"
"""The key of this tool's state in :attr:`Session.state`: ``{"labels": [...], "status":
..., "undecided": ..., "trace_id": ...}`` for the session's last classified turn."""


class Classifier:
    """Assigns labels from a fixed label set to a text.

    ``multi_label=False`` (default) chooses exactly one label or none; ``True`` chooses
    every label that applies. Either way a text the classifier cannot settle comes back
    ``UNDECIDED`` -- see :class:`~kuhaku.tools.classification.result.UndecidedReason`.

    The default pipeline is ``rules`` (only when ``rules`` are given) followed by
    ``classify``; ``steps`` are appended after those. Compose further through
    :attr:`pipeline` -- ``add(..., before=/after=)``, ``remove``, ``replace``,
    ``enable``/``disable``.

    ``min_confidence`` is the confidence a model-proposed label needs to be chosen.
    ``instructions`` is the caller-owned part of the system prompt (domain rules, e.g.
    "a question about an invoice is billing, not account"), added after the framework's
    own rules, never replacing them.

    ``llm`` takes any :class:`~kuhaku.core.llm.base.LLMProvider`; when omitted, the
    provider is built from ``settings`` (or the environment) through
    :func:`~kuhaku.core.llm.build_llm_provider`, the same path ``RAG`` uses.
    ``fallback_llms`` are tried in order when ``llm`` does not reach a decision -- see
    :class:`~kuhaku.tools.classification.steps.ClassifyStep`. Every provider is wrapped
    in :class:`~kuhaku.core.llm.token_tracking.TokenTrackingLLM` unless
    ``enable_token_tracking=False``, as ``RAG`` does.

    ``native_structured_output`` lets the classify step use a provider's native
    structured-output mode when it has one (the default); ``False`` always uses the
    prompt alone.

    ``prompts`` is a :class:`~kuhaku.core.prompts.PromptRegistry` layered over the
    built-in templates (``classification.system``, ``classification.single_label``,
    ``classification.multi_label``, ``classification.user``, ``classification.repair``,
    and for attachments ``classification.attachment_rule``, ``classification.attachments``,
    ``classification.user_attachments``): a name registered there replaces kuhaku's
    wording. ``result.metadata["prompts"]`` reports which name, version and hash each
    role of the deciding call used.

    ``session_window`` picks which of a session's messages are shown as history when
    :meth:`classify` is given ``session=``; the default is the last 10 turns.
    """

    def __init__(
        self,
        labels: LabelSet | Iterable[Label | str],
        *,
        multi_label: bool = False,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
        rules: Sequence[Rule] = (),
        steps: Sequence[Step] = (),
        instructions: str = "",
        min_confidence: float = 0.5,
        max_repair_attempts: int = 1,
        fallback_llms: Sequence[LLMProvider] = (),
        enable_token_tracking: bool = True,
        native_structured_output: bool = True,
        prompts: PromptRegistry | None = None,
        session_window: ContextWindow | None = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ConfigError(
                f"min_confidence must be between 0 and 1; got {min_confidence}."
            )
        self.labels = as_label_set(labels)
        self.multi_label = multi_label
        self.min_confidence = min_confidence
        self.prompts: PromptRegistry = layered(prompts)
        self.session_window = session_window or ContextWindow(max_turns=10)
        built = (
            llm if llm is not None else build_llm_provider(settings or get_settings())
        )
        track = _token_tracked if enable_token_tracking else _untracked
        self.llm: LLMProvider = track(built)
        fallbacks = [track(fallback) for fallback in fallback_llms]
        self.pipeline = Pipeline(
            [
                *([RuleStep(*rules)] if rules else []),
                ClassifyStep(
                    instructions=instructions,
                    max_repair_attempts=max_repair_attempts,
                    fallbacks=fallbacks,
                    native_structured_output=native_structured_output,
                ),
                *steps,
            ]
        )

    def classify(
        self,
        text: str | Document | None = None,
        *,
        attachments: Sequence[MediaPart] = (),
        history: Sequence[Message] = (),
        trace_id: str | None = None,
        session: Session | None = None,
    ) -> ClassificationResult:
        """Classify ``text``, ``attachments``, or both.

        ``text`` may also be a paged document -- anything satisfying
        :class:`~kuhaku.tools.classification.documents.Document`, such as the parsing tool's
        ``ParsedDocument``; its pages are rendered as Markdown and joined, and
        ``result.metadata["document"]`` names the source and pages.

        ``attachments`` are images and PDFs (:class:`~kuhaku.core.media.ImagePart`,
        :class:`~kuhaku.core.media.DocumentPart`) that are part of what is classified.
        They need a provider whose ``capabilities.content_types`` includes their type;
        fallbacks that cannot take them are skipped, and with none that can,
        :class:`~kuhaku.core.llm.base.UnsupportedContentError` is raised. Rules match
        ``text`` only.

        ``history`` (earlier conversation turns) is shown to the model only to resolve
        what ``text`` refers to -- "yes, cancel it" is a cancellation only in context.
        Rules see it on ``ctx.history`` but the built-in ones match ``text`` alone.

        ``session`` replaces passing ``history`` by hand: the history is
        :attr:`session_window` applied to the session's messages; afterwards ``text``
        is appended as a user message and the outcome is written to the session's
        ``"classification"`` state. The session is only changed in memory -- persist it
        with ``SessionStore.save``. Passing both ``session`` and ``history`` is an
        error. If the call raises, the session is left untouched.

        ``trace_id`` binds every log record and span of this call to an id; without
        one, an id bound by the caller is inherited, else a new one is minted.
        """

        if text is None:
            text = ""
        document: dict[str, Any] | None = None
        if is_document(text):
            flat = document_text(text)  # type: ignore[arg-type]
            text = flat.text
            document = {
                "source": flat.source,
                "pages": [number for number, _ in flat.pages],
            }
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}.")
        if isinstance(attachments, MediaPart):
            attachments = (attachments,)
        attachments = tuple(attachments)
        for part in attachments:
            if not isinstance(part, MediaPart):
                raise TypeError(
                    f"attachments must be ImagePart/DocumentPart objects; got "
                    f"{type(part).__name__}."
                )
        if session is not None:
            if history:
                raise ConfigError(
                    "Pass session= or history=, not both: with a session, the history "
                    "comes from the session."
                )
            history = self.session_window.apply(session.messages)
        span_fields: dict[str, Any] = {
            "labels": len(self.labels),
            "multi_label": self.multi_label,
        }
        if session is not None:
            span_fields["session_id"] = loggable_session_id(session.id)
        if attachments:
            span_fields["attachments"] = len(attachments)
        with (
            bind_trace_id(trace_id) as tid,
            instrumented_step("classification", logger, **span_fields) as rec,
        ):
            ctx = ClassificationContext(
                labels=self.labels,
                text=text,
                llm=self.llm,
                multi_label=self.multi_label,
                min_confidence=self.min_confidence,
                history=tuple(history),
                trace_id=tid,
                attachments=attachments,
                prompts=self.prompts,
            )
            if document is not None:
                ctx.metadata["document"] = document
            self.pipeline.run(ctx)
            result = ctx.to_result()
            rec.set(
                outcome=result.status.value,
                decided_by=result.decided_by,
                undecided=result.undecided.value if result.undecided else None,
                attempts=result.attempts,
            )
            if session is not None:
                self._record_turn(session, text, result, len(attachments))
            return result

    def _record_turn(
        self,
        session: Session,
        text: str,
        result: ClassificationResult,
        attachments: int,
    ) -> None:
        metadata: dict[str, Any] = {
            "tool": SESSION_NAMESPACE,
            "trace_id": result.trace_id,
        }
        if attachments:
            # Media bytes do not belong in a session record; the count says they existed.
            metadata["attachments"] = attachments
        session.append(Message("user", text, metadata))
        state = session.tool_state(SESSION_NAMESPACE)
        state.clear()
        state.update(
            {
                "labels": list(result.labels),
                "status": result.status.value,
                "undecided": result.undecided.value if result.undecided else None,
                "trace_id": result.trace_id,
            }
        )

    def evaluate_sample(self, query: str) -> EvaluationSample:
        """Implements ``EvaluationTarget`` (see ``kuhaku.evaluation.target``).

        ``answer`` is the chosen labels joined by spaces (``None`` when undecided), so
        ``AnswerCorrectnessMetric``'s token overlap against a golden answer of label
        names is exact match for single-label and Jaccard over label sets for
        multi-label -- provided label names contain no spaces. The full outcome is in
        ``metadata``.
        """

        result = self.classify(query)
        return EvaluationSample(
            query=query,
            answer=" ".join(result.labels) if result.labels else None,
            metadata={
                "trace_id": result.trace_id,
                "status": result.status.value,
                "labels": list(result.labels),
                "scores": dict(result.scores),
                "decided_by": result.decided_by,
                "undecided": result.undecided.value if result.undecided else None,
            },
        )


def _token_tracked(llm: LLMProvider) -> LLMProvider:
    if isinstance(llm, TokenTrackingLLM):
        return llm
    # A provider's name is "<provider>:<model>"; the part before the colon is what
    # TokenTrackingLLM labels its metrics with.
    return TokenTrackingLLM(llm, provider=llm.name.split(":", 1)[0])


def _untracked(llm: LLMProvider) -> LLMProvider:
    return llm
