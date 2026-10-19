"""Faithfulness metric: is the generated answer supported by the retrieved context?

Optional and disabled by default -- only active when a judge is supplied, matching
``judges.py``'s own "opt-in everywhere" stance. ``EvaluationRunner`` may auto-create a
judge and inject it via the ``judge`` setter when none was supplied explicitly.
"""

from __future__ import annotations

import logging

from ..base_metric import BaseMetric
from ..judges import LLMFaithfulnessEvaluator
from ..sample import EvaluationSample

logger = logging.getLogger("kuhaku.evaluation.metrics.faithfulness")


class FaithfulnessMetric(BaseMetric):
    """Scores answer faithfulness via an ``LLMFaithfulnessEvaluator``.

    Returns ``{}`` (skip) when no judge is configured or no answer was generated. A
    judge failure is caught, logged, and also skipped rather than crashing the run --
    the judge itself already has its own keyword-overlap fallback, but a custom judge
    implementation might not.

    When the judge reports that its score came from the keyword-overlap fallback (see
    ``LLMFaithfulnessEvaluator.evaluate_detailed``), the score is recorded as
    ``faithfulness_fallback`` instead of ``faithfulness``: the two measure different
    things, and mixing them would make a run whose judge was down look like a
    faithfulness change.
    """

    name = "faithfulness"

    def __init__(self, judge: LLMFaithfulnessEvaluator | None = None, threshold: float = 0.5) -> None:
        self._judge = judge
        self._threshold = threshold

    @property
    def judge(self) -> LLMFaithfulnessEvaluator | None:
        return self._judge

    @judge.setter
    def judge(self, value: LLMFaithfulnessEvaluator | None) -> None:
        self._judge = value

    def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
        if self._judge is None or sample.answer is None:
            return {}
        detailed = getattr(self._judge, "evaluate_detailed", None)
        try:
            if callable(detailed):
                result = detailed(sample.query, sample.contexts or [], sample.answer)
                key = "faithfulness_fallback" if result.fallback else "faithfulness"
                return {key: result.score}
            score = self._judge.evaluate(sample.query, sample.contexts or [], sample.answer)
        except Exception as exc:
            logger.warning("faithfulness judge failed for query=%r: %s", sample.query, exc)
            return {}
        return {"faithfulness": score}
