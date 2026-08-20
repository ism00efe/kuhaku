"""Retrieval-quality metrics: how good is the retrieved id set, independent of any
generated answer.

Score against ``sample.retrieved_doc_ids`` (what the target returned) and
``sample.relevant_doc_ids`` (the golden set) -- never ``sample.contexts``, which is text,
not stable identifiers.
"""

from __future__ import annotations

from ..base_metric import BaseMetric
from ..sample import EvaluationSample
from . import functions as fn


class RecallAtKMetric(BaseMetric):
    name = "recall_at_k"

    def __init__(self, k: int = 5) -> None:
        self._k = k

    def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
        score = fn.recall_at_k(
            sample.retrieved_doc_ids or [], sample.relevant_doc_ids or set(), self._k
        )
        return {"recall_at_k": score}


class PrecisionAtKMetric(BaseMetric):
    name = "precision_at_k"

    def __init__(self, k: int = 5) -> None:
        self._k = k

    def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
        score = fn.precision_at_k(
            sample.retrieved_doc_ids or [], sample.relevant_doc_ids or set(), self._k
        )
        return {"precision_at_k": score}


class MRRMetric(BaseMetric):
    name = "mrr"

    def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
        score = fn.mrr(sample.retrieved_doc_ids or [], sample.relevant_doc_ids or set())
        return {"mrr": score}


class NDCGAtKMetric(BaseMetric):
    name = "ndcg_at_k"

    def __init__(self, k: int = 5) -> None:
        self._k = k

    def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
        score = fn.ndcg_at_k(
            sample.retrieved_doc_ids or [], sample.relevant_doc_ids or set(), self._k
        )
        return {"ndcg_at_k": score}


class HitRateAtKMetric(BaseMetric):
    name = "hit_rate_at_k"

    def __init__(self, k: int = 5) -> None:
        self._k = k

    def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
        score = fn.hit_rate_at_k(
            sample.retrieved_doc_ids or [], sample.relevant_doc_ids or set(), self._k
        )
        return {"hit_rate_at_k": score}
