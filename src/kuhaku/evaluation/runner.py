"""``EvaluationRunner``: the central orchestrator for a tool-agnostic evaluation run.

Accepts any ``EvaluationTarget`` (or an object/callable ``TargetAdapter`` can wrap into
one -- see ``target.py``), runs it over a golden dataset, merges each produced
``EvaluationSample`` with that question's golden fields, and scores it with a list of
pluggable ``BaseMetric`` instances. Adding a metric never touches this file; neither does
swapping the target's shape -- all generic target adaptation lives in ``TargetAdapter``,
not here. This runner has no knowledge of any specific tool (RAG or otherwise): to use a
tool-specific adapter (e.g. RAG's ``RAGTargetAdapter``, with its ``llm_provider``
fallback), construct it explicitly and pass the already-built adapter as ``target`` --
``EvaluationTarget`` instances are used as-is, no further adaptation.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable

from kuhaku.core.config import Settings
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.base import LLMProvider

from .base_metric import BaseMetric
from .dataset import load_eval_dataset
from .judges import LLMFaithfulnessEvaluator
from .metrics import FaithfulnessMetric
from .sample import EvaluationSample
from .store import EvaluationStore, InMemoryEvaluationStore
from .target import EvaluationTarget, TargetAdapter

logger = logging.getLogger("kuhaku.evaluation.runner")


class EvaluationRunner:
    """Runs a golden dataset through a target, scoring and storing each question."""

    def __init__(
        self,
        metrics: list[BaseMetric],
        dataset_path: str | Path,
        *,
        store: EvaluationStore | None = None,
        run_id: str | None = None,
        judge_llm_provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._metrics = metrics
        self._dataset_path = dataset_path
        self._store = store if store is not None else InMemoryEvaluationStore()
        self._run_id = run_id if run_id is not None else uuid.uuid4().hex
        self._judge = self._build_judge(judge_llm_provider, settings)

    @staticmethod
    def _build_judge(
        judge_llm_provider: LLMProvider | None, settings: Settings | None
    ) -> LLMFaithfulnessEvaluator | None:
        """Explicit injection wins. Otherwise, given ``settings``, auto-create a
        zero-temperature judge provider (a judge must be deterministic,
        independent of whatever temperature the live target's own provider runs at).
        With neither, stays ``None`` -- faithfulness scoring is opt-in everywhere in this
        package (see ``judges.py``), never a hard requirement to construct a runner."""

        if judge_llm_provider is not None:
            return LLMFaithfulnessEvaluator(judge_llm_provider)
        if settings is None:
            return None
        try:
            provider = build_llm_provider(settings.model_copy(update={"llm_temperature": 0.0}))
        except Exception as exc:
            logger.warning("could not auto-create judge LLM provider: %s", exc)
            return None
        return LLMFaithfulnessEvaluator(provider)

    def _wire_judge(self) -> None:
        """Inject the runner's judge into any ``FaithfulnessMetric`` that wasn't given
        one explicitly -- explicit injection at metric-construction time always wins."""

        if self._judge is None:
            return
        for metric in self._metrics:
            if isinstance(metric, FaithfulnessMetric) and metric.judge is None:
                metric.judge = self._judge

    def run(
        self,
        target: Any,
        *,
        top_k: int = 5,
        answer_generator: Callable[[str, list[Any]], str | None] | None = None,
    ) -> dict[str, float]:
        eval_target = self._adapt(target, top_k=top_k, answer_generator=answer_generator)
        self._wire_judge()

        dataset = load_eval_dataset(self._dataset_path)

        for item in dataset:
            try:
                sample = eval_target.evaluate_sample(item.question)
            except Exception as exc:
                logger.warning(
                    "target failed for question_id=%s: %s", item.question_id, exc
                )
                sample = EvaluationSample(query=item.question)

            sample = sample.with_golden(
                relevant_doc_ids=set(item.expected_sources),
                metadata={
                    "question_id": item.question_id,
                    "golden_answer": item.golden_answer,
                    "category": item.category,
                },
            )

            metrics_result: dict[str, float] = {}
            for metric in self._metrics:
                try:
                    result = metric.evaluate(sample)
                except Exception as exc:
                    logger.warning(
                        "metric %s failed for question_id=%s: %s",
                        getattr(metric, "name", type(metric).__name__),
                        item.question_id,
                        exc,
                    )
                    continue
                if result:
                    metrics_result.update(result)

            self._store.add_result(
                item.question_id,
                sample.retrieved_doc_ids or [],
                metrics_result,
                sample.answer,
                run_id=self._run_id,
                metadata={"category": item.category} if item.category is not None else None,
            )

        return self._store.summary(run_id=self._run_id)

    @staticmethod
    def _adapt(
        target: Any,
        *,
        top_k: int,
        answer_generator: Callable[[str, list[Any]], str | None] | None,
    ) -> EvaluationTarget:
        if isinstance(target, EvaluationTarget):
            return target
        return TargetAdapter(target, top_k=top_k, answer_generator=answer_generator)
