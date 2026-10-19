from __future__ import annotations

import pytest

from kuhaku.evaluation.judges import LLMFaithfulnessEvaluator, parse_judge_score
from kuhaku.evaluation.metrics import FaithfulnessMetric
from kuhaku.evaluation.sample import EvaluationSample


class FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, system: str, user: str) -> str:
        return self._response


class RaisingLLM:
    def generate(self, system: str, user: str) -> str:
        raise RuntimeError("provider unavailable")


def test_evaluate_parses_numeric_score_from_llm():
    evaluator = LLMFaithfulnessEvaluator(FakeLLM("0.8"))
    score = evaluator.evaluate("Q?", ["some context"], "an answer")
    assert score == 0.8


def test_evaluate_clamps_out_of_range_score():
    evaluator = LLMFaithfulnessEvaluator(FakeLLM("5.0"))
    assert evaluator.evaluate("Q?", ["context"], "answer") == 1.0

    evaluator = LLMFaithfulnessEvaluator(FakeLLM("-2.0"))
    assert evaluator.evaluate("Q?", ["context"], "answer") == 0.0


def test_evaluate_falls_back_to_keyword_overlap_when_provider_raises():
    evaluator = LLMFaithfulnessEvaluator(RaisingLLM())
    score = evaluator.evaluate("Q?", ["the quick brown fox"], "the quick brown fox")
    assert score == 1.0


def test_evaluate_falls_back_when_response_has_no_number():
    evaluator = LLMFaithfulnessEvaluator(FakeLLM("I cannot answer that"))
    score = evaluator.evaluate("Q?", ["completely unrelated text"], "totally different words")
    assert score == 0.0


def test_fallback_with_no_context_is_zero():
    evaluator = LLMFaithfulnessEvaluator(RaisingLLM())
    score = evaluator.evaluate("Q?", [], "some answer")
    assert score == 0.0


def test_parse_takes_last_number_when_judge_restates_the_scale():
    assert parse_judge_score("On a 0 to 1 scale, I rate this 0.9") == pytest.approx(0.9)


def test_parse_prefers_json_score():
    assert parse_judge_score('Scale 0-1. {"score": 0.4} (confident: 1)') == pytest.approx(0.4)


def test_evaluate_detailed_marks_fallback():
    evaluator = LLMFaithfulnessEvaluator(RaisingLLM())
    result = evaluator.evaluate_detailed("Q?", ["a b"], "a b")
    assert result.fallback is True
    judged = LLMFaithfulnessEvaluator(FakeLLM("0.7")).evaluate_detailed("Q?", [], "a")
    assert judged.fallback is False


def test_metric_records_fallback_under_its_own_key():
    metric = FaithfulnessMetric(judge=LLMFaithfulnessEvaluator(RaisingLLM()))
    result = metric.evaluate(EvaluationSample(query="Q?", answer="a b", contexts=["a b"]))
    assert result == {"faithfulness_fallback": 1.0}

    metric = FaithfulnessMetric(judge=LLMFaithfulnessEvaluator(FakeLLM('{"score": 0.6}')))
    result = metric.evaluate(EvaluationSample(query="Q?", answer="a", contexts=["a"]))
    assert result == {"faithfulness": 0.6}
