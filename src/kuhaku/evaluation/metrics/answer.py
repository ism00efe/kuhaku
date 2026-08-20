"""Answer-quality metrics: how good is the generated answer text, given a golden label."""

from __future__ import annotations

from ..base_metric import BaseMetric
from ..sample import EvaluationSample
from . import functions as fn


class AnswerCorrectnessMetric(BaseMetric):
    """Token-overlap (Jaccard) similarity against ``sample.metadata["golden_answer"]``.

    A baseline fallback only -- prefer ``FaithfulnessMetric`` with an LLM judge for a
    semantic score. Skips (returns ``{}``) when no answer was generated, or no golden
    answer is present in ``metadata`` (a non-RAG target with nothing to compare against).
    """

    name = "answer_correctness"

    def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
        golden_answer = sample.metadata.get("golden_answer")
        if sample.answer is None or golden_answer is None:
            return {}
        score = fn.answer_correctness(sample.answer, golden_answer)
        return {"answer_correctness": score}
