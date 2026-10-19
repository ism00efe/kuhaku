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

from kuhaku.core.batch import BatchProgress, BatchRunner
from kuhaku.core.config import Settings
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.base import LLMProvider

from .base_metric import BaseMetric
from .dataset import EvaluationItem, load_eval_dataset
from .judges import LLMFaithfulnessEvaluator
from .metrics import FaithfulnessMetric
from .sample import EvaluationSample
from .store import EvaluationStore, InMemoryEvaluationStore
from .target import EvaluationTarget, TargetAdapter

logger = logging.getLogger("kuhaku.evaluation.runner")

_Scored = tuple[EvaluationItem, EvaluationSample, dict[str, float]]


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
        concurrency: int = 1,
        on_progress: Callable[[BatchProgress], None] | None = None,
    ) -> dict[str, float]:
        """Score every dataset question; returns the per-metric mean for this run.

        ``concurrency > 1`` (or an ``on_progress`` hook) evaluates questions -- target
        call and metric scoring -- through :class:`~kuhaku.core.batch.BatchRunner`; with
        more than one worker the target and metrics must be safe to call from several
        threads. Store writes always happen on the calling thread, in dataset order, so
        the store receives exactly what a sequential run would have written.
        """

        eval_target = self._adapt(target, top_k=top_k, answer_generator=answer_generator)
        self._wire_judge()

        dataset = load_eval_dataset(self._dataset_path)

        if concurrency == 1 and on_progress is None:
            # The default stays a plain loop that stores each row as soon as it is
            # scored: no worker threads, no per-item trace id (the target's log lines
            # keep the caller's), and an interrupted run keeps the rows it finished.
            for item in dataset:
                self._store_scored(self._evaluate_item(eval_target, item))
            return self._store.summary(run_id=self._run_id)

        runner: BatchRunner[EvaluationItem, _Scored] = BatchRunner(
            concurrency=concurrency, on_progress=on_progress, name="evaluation"
        )
        outcomes = runner.run(dataset, lambda item: self._evaluate_item(eval_target, item))
        # _evaluate_item already catches every target and metric failure, so a failed
        # outcome is a bug in this module -- surface it rather than store a hole.
        for outcome in outcomes:
            if not outcome.success:
                raise RuntimeError(f"evaluation item failed unexpectedly: {outcome.error}")
        for outcome in outcomes:
            self._store_scored(outcome.output)
        return self._store.summary(run_id=self._run_id)

    def _store_scored(self, scored: _Scored) -> None:
        item, sample, metrics_result = scored
        metadata: dict[str, Any] = {}
        if item.category is not None:
            metadata["category"] = item.category
        # A target that raised still gets a row (its scores are what the run really
        # achieved), but the row says why -- otherwise it reads as a bad answer.
        if sample.metadata.get("error"):
            metadata["error"] = sample.metadata["error"]
        self._store.add_result(
            item.question_id,
            sample.retrieved_doc_ids or [],
            metrics_result,
            sample.answer,
            run_id=self._run_id,
            metadata=metadata or None,
        )

    def _evaluate_item(self, eval_target: EvaluationTarget, item: EvaluationItem) -> _Scored:
        try:
            sample = eval_target.evaluate_sample(item.question)
        except Exception as exc:
            logger.warning("target failed for question_id=%s: %s", item.question_id, exc)
            sample = EvaluationSample(
                query=item.question, metadata={"error": f"{type(exc).__name__}: {exc}"}
            )

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
        return item, sample, metrics_result

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
