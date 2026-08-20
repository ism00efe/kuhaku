"""``EvaluationTarget``: the tool-agnostic contract ``EvaluationRunner`` scores.

``TargetAdapter`` is the only place that still knows about generic agent/tool shapes
(``retrieve()``/``search()``, ``ask()``/``answer()``) -- it exists purely to keep
``EvaluationRunner`` itself free of that knowledge, translating anything it is handed
into ``EvaluationSample`` objects. It carries no RAG-specific behavior (no prompt
templates, no RAG type): a tool that wants a domain-flavored answer-generation fallback
(e.g. RAG's ``llm_provider``-driven default prompting) should subclass it -- see
``kuhaku.tools.rag.evaluation_target.RAGTargetAdapter``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

from ._extraction import extract_document_ids, extract_texts
from .sample import EvaluationSample

logger = logging.getLogger("kuhaku.evaluation.target")


@runtime_checkable
class EvaluationTarget(Protocol):
    """Anything that can answer a query and describe what it did as an ``EvaluationSample``."""

    def evaluate_sample(self, query: str) -> EvaluationSample: ...


class TargetAdapter:
    """Wraps a callable or an existing object into an ``EvaluationTarget``.

    Supports, in order of preference:

    1. An object that already implements ``evaluate_sample(query) -> EvaluationSample``
       (used as-is -- wrapping it again would be a no-op indirection).
    2. A plain callable ``query -> EvaluationSample`` or ``query -> str | None`` (a bare
       answer string, wrapped into a sample with no context).
    3. An object exposing ``ask()``/``answer()`` -- its return value's
       ``.text``/``.retrieved`` are read the way a RAG-style engine's results always
       have been (duck-typed, not tied to any specific RAG type).
    4. A bare retriever exposing ``retrieve()``/``search()`` -- context-only; an answer
       is generated via ``answer_generator`` if supplied, else left unset.
    """

    def __init__(
        self,
        target: Any,
        *,
        top_k: int = 5,
        answer_generator: Callable[[str, list[Any]], str | None] | None = None,
    ) -> None:
        self._target = target
        self._top_k = top_k
        self._answer_generator = answer_generator
        self._mode, self._fn = self._resolve(target)

    @staticmethod
    def _resolve(target: Any) -> tuple[str, Callable[..., Any]]:
        if hasattr(target, "evaluate_sample") and callable(target.evaluate_sample):
            return "native", target.evaluate_sample

        ask_fn = getattr(target, "ask", None) or getattr(target, "answer", None)
        retrieve_fn = getattr(target, "retrieve", None) or getattr(target, "search", None)
        if ask_fn is not None:
            return "ask", ask_fn
        if retrieve_fn is not None:
            return "retrieve", retrieve_fn

        if callable(target):
            return "callable", target

        raise TypeError(
            f"target must implement evaluate_sample(), retrieve(), search(), ask(), or "
            f"answer(), or be callable; got {type(target).__name__}"
        )

    def evaluate_sample(self, query: str) -> EvaluationSample:
        if self._mode == "native":
            return self._fn(query)
        if self._mode == "callable":
            return self._evaluate_callable(query)
        if self._mode == "ask":
            return self._evaluate_ask(query)
        return self._evaluate_retrieve(query)

    def _evaluate_callable(self, query: str) -> EvaluationSample:
        result = self._fn(query)
        if isinstance(result, EvaluationSample):
            return result
        return EvaluationSample(query=query, answer=result if result is not None else None)

    def _evaluate_ask(self, query: str) -> EvaluationSample:
        try:
            ask_result = self._fn(query)
        except Exception as exc:
            logger.warning("target call failed for query=%r: %s", query, exc)
            return EvaluationSample(query=query)

        retrieved = list(getattr(ask_result, "retrieved", None) or [])
        answer = self._generate_answer(
            query, retrieved, default=getattr(ask_result, "text", None), allow_fallback=False
        )
        return EvaluationSample(
            query=query,
            answer=answer,
            contexts=extract_texts(retrieved) if retrieved else [],
            retrieved_doc_ids=self._extract_ids(retrieved),
        )

    def _evaluate_retrieve(self, query: str) -> EvaluationSample:
        try:
            retrieved = self._fn(query, self._top_k)
        except Exception as exc:
            logger.warning("retrieve failed for query=%r: %s", query, exc)
            return EvaluationSample(query=query, contexts=[], retrieved_doc_ids=[])

        answer = self._generate_answer(query, retrieved, default=None)
        return EvaluationSample(
            query=query,
            answer=answer,
            contexts=extract_texts(retrieved) if retrieved else [],
            retrieved_doc_ids=self._extract_ids(retrieved),
        )

    def _generate_answer(
        self,
        query: str,
        retrieved: list[Any],
        *,
        default: str | None,
        allow_fallback: bool = True,
    ) -> str | None:
        """``answer_generator`` (given the raw ``retrieved`` items, matching the pre-
        tool-agnostic contract) wins if supplied. Otherwise ``default`` -- an ``ask()``
        target's own answer text, when there is one -- wins.

        A subclass that wants a further, domain-specific fallback (e.g. RAG's
        ``llm_provider``-driven default prompting) should override this method, calling
        ``super()._generate_answer(...)`` first and only falling back when it returns
        ``None`` and ``allow_fallback`` is ``True`` -- see
        ``RAGTargetAdapter._generate_answer``.
        """

        if self._answer_generator is not None:
            try:
                return self._answer_generator(query, retrieved)
            except Exception as exc:
                logger.warning("answer_generator failed for query=%r: %s", query, exc)
                return None
        if default is not None or not allow_fallback:
            return default
        return None

    @staticmethod
    def _extract_ids(retrieved: list[Any]) -> list[str]:
        try:
            return extract_document_ids(retrieved)
        except TypeError as exc:
            logger.warning("could not extract document ids: %s", exc)
            return []
