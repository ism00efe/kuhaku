from __future__ import annotations

import pytest

from kuhaku.evaluation.metrics import (
    AnswerCorrectnessMetric,
    FaithfulnessMetric,
    HitRateAtKMetric,
    MRRMetric,
    NDCGAtKMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
)
from kuhaku.evaluation.sample import EvaluationSample


def _sample(**overrides) -> EvaluationSample:
    defaults: dict = dict(query="Q?")
    defaults.update(overrides)
    return EvaluationSample(**defaults)


# -- retrieval metrics --------------------------------------------------------------


def test_recall_at_k_metric_hit():
    metric = RecallAtKMetric(k=2)
    result = metric.evaluate(
        _sample(retrieved_doc_ids=["a", "b"], relevant_doc_ids={"a"})
    )
    assert result == {"recall_at_k": 1.0}


def test_recall_at_k_metric_empty_retrieved_is_zero():
    metric = RecallAtKMetric(k=5)
    result = metric.evaluate(_sample(retrieved_doc_ids=[], relevant_doc_ids={"a"}))
    assert result == {"recall_at_k": 0.0}


def test_recall_at_k_metric_no_retrieved_doc_ids_is_zero():
    metric = RecallAtKMetric(k=5)
    result = metric.evaluate(_sample(relevant_doc_ids={"a"}))
    assert result == {"recall_at_k": 0.0}


def test_precision_at_k_metric():
    metric = PrecisionAtKMetric(k=2)
    result = metric.evaluate(
        _sample(retrieved_doc_ids=["a", "x"], relevant_doc_ids={"a"})
    )
    assert result == {"precision_at_k": 0.5}


def test_mrr_metric():
    metric = MRRMetric()
    result = metric.evaluate(
        _sample(retrieved_doc_ids=["x", "a"], relevant_doc_ids={"a"})
    )
    assert result == {"mrr": 0.5}


def test_ndcg_at_k_metric_perfect_ranking():
    metric = NDCGAtKMetric(k=2)
    result = metric.evaluate(_sample(retrieved_doc_ids=["a"], relevant_doc_ids={"a"}))
    assert result == {"ndcg_at_k": 1.0}


def test_hit_rate_at_k_metric_miss():
    metric = HitRateAtKMetric(k=2)
    result = metric.evaluate(
        _sample(retrieved_doc_ids=["x", "y"], relevant_doc_ids={"a"})
    )
    assert result == {"hit_rate_at_k": 0.0}


# -- answer correctness --------------------------------------------------------------


def test_answer_correctness_metric_none_answer_skips():
    metric = AnswerCorrectnessMetric()
    assert metric.evaluate(_sample(metadata={"golden_answer": "golden"})) == {}


def test_answer_correctness_metric_no_golden_answer_skips():
    metric = AnswerCorrectnessMetric()
    assert metric.evaluate(_sample(answer="a b c")) == {}


def test_answer_correctness_metric_computes_jaccard():
    metric = AnswerCorrectnessMetric()
    result = metric.evaluate(
        _sample(answer="a b c", metadata={"golden_answer": "a b d"})
    )
    assert result["answer_correctness"] == pytest.approx(2 / 4)


# -- faithfulness ----------------------------------------------------------------------


class StubJudge:
    def __init__(self, score: float = 0.75) -> None:
        self.calls: list[tuple[str, list[str], str]] = []
        self._score = score

    def evaluate(self, question: str, context_chunks: list[str], generated_answer: str) -> float:
        self.calls.append((question, context_chunks, generated_answer))
        return self._score


class FailingJudge:
    def evaluate(self, question: str, context_chunks: list[str], generated_answer: str) -> float:
        raise RuntimeError("boom")


def test_faithfulness_metric_no_judge_skips():
    metric = FaithfulnessMetric(judge=None)
    assert metric.evaluate(_sample(answer="an answer")) == {}


def test_faithfulness_metric_no_answer_skips():
    metric = FaithfulnessMetric(judge=StubJudge())
    assert metric.evaluate(_sample()) == {}


def test_faithfulness_metric_scores_with_judge():
    judge = StubJudge(score=0.75)
    metric = FaithfulnessMetric(judge=judge)
    result = metric.evaluate(_sample(answer="an answer", contexts=["context text"]))
    assert result == {"faithfulness": 0.75}
    assert judge.calls == [("Q?", ["context text"], "an answer")]


def test_faithfulness_metric_judge_failure_skips_and_does_not_raise():
    metric = FaithfulnessMetric(judge=FailingJudge())
    assert metric.evaluate(_sample(answer="an answer")) == {}


def test_faithfulness_metric_judge_setter_overrides_when_none():
    metric = FaithfulnessMetric()
    assert metric.judge is None
    judge = StubJudge(score=0.5)
    metric.judge = judge
    result = metric.evaluate(_sample(answer="an answer", contexts=["ctx"]))
    assert result == {"faithfulness": 0.5}


def test_faithfulness_metric_explicit_judge_not_overwritten():
    explicit = StubJudge(score=0.9)
    metric = FaithfulnessMetric(judge=explicit)
    other = StubJudge(score=0.1)
    if metric.judge is None:
        metric.judge = other
    result = metric.evaluate(_sample(answer="an answer", contexts=["ctx"]))
    assert result == {"faithfulness": 0.9}
