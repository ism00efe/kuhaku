from __future__ import annotations

from kuhaku.evaluation.judges import LLMFaithfulnessEvaluator


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
