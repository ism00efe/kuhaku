"""Measuring extraction with kuhaku's own evaluation runner."""

from __future__ import annotations

import json

from kuhaku.evaluation import (
    EvaluationRunner,
    EvaluationSample,
    EvaluationTarget,
    StructuredMatchMetric,
)
from kuhaku.tools.extraction import ExtractionRouter, Extractor

from .conftest import Complaint, Order, ScriptedLLM


def _sample(answer, golden):
    return EvaluationSample(
        query="q",
        answer=None if answer is None else json.dumps(answer),
        metadata={"golden_answer": json.dumps(golden)},
    )


def test_field_accuracy_and_exact_match():
    metric = StructuredMatchMetric()
    golden = {"product": "tea", "quantity": 5, "unit": None}

    assert metric.evaluate(_sample({"product": "Tea ", "quantity": 5.0}, golden)) == {
        "field_accuracy": 1.0,
        "exact_match": 1.0,
    }
    half = metric.evaluate(_sample({"product": "coffee", "quantity": 5, "unit": "kg"}, golden))
    assert half["field_accuracy"] == 1 / 3
    assert half["exact_match"] == 0.0


def test_nested_lists_are_compared_by_pointer():
    golden = {"lines": [{"item": "tea"}, {"item": "sugar"}]}
    answer = {"lines": [{"item": "tea"}, {"item": "salt"}]}
    assert StructuredMatchMetric().evaluate(_sample(answer, golden))["field_accuracy"] == 0.5


def test_a_missing_answer_scores_zero_and_a_non_json_label_is_skipped():
    metric = StructuredMatchMetric()
    assert metric.evaluate(_sample(None, {"a": 1})) == {"field_accuracy": 0.0, "exact_match": 0.0}
    sample = EvaluationSample(query="q", answer="{}", metadata={"golden_answer": "plain text"})
    assert metric.evaluate(sample) == {}


def test_strict_string_comparison_can_be_chosen():
    metric = StructuredMatchMetric(normalize_strings=False)
    assert metric.evaluate(_sample({"a": "Tea"}, {"a": "tea"}))["exact_match"] == 0.0


def test_an_extractor_is_an_evaluation_target():
    llm = ScriptedLLM({"data": {"product": "tea", "quantity": None}})
    extractor = Extractor(Order, llm=llm)
    assert isinstance(extractor, EvaluationTarget)

    sample = extractor.evaluate_sample("tea please")
    assert json.loads(sample.answer) == {"product": "tea", "quantity": None}
    assert sample.metadata["status"] == "needs_clarification"
    assert sample.metadata["missing_slots"] == ["/quantity"]


def test_the_runner_scores_an_extractor_end_to_end(tmp_path):
    dataset = tmp_path / "golden.jsonl"
    rows = [
        {"question_id": "1", "question": "5 tea", "golden_answer": json.dumps({"product": "tea", "quantity": 5})},
        {"question_id": "2", "question": "2 sugar", "golden_answer": json.dumps({"product": "sugar", "quantity": 2})},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    llm = ScriptedLLM(
        {"data": {"product": "tea", "quantity": 5}},
        {"data": {"product": "salt", "quantity": 2}},
    )

    summary = EvaluationRunner([StructuredMatchMetric()], dataset).run(Extractor(Order, llm=llm))

    assert summary["field_accuracy"] == 0.75
    assert summary["exact_match"] == 0.5


def test_a_router_answer_checks_the_routing_too():
    router = ExtractionRouter(
        [Extractor(Order, llm=ScriptedLLM()), Extractor(Complaint, llm=ScriptedLLM({"data": {"topic": "late"}}))],
        llm=ScriptedLLM({"data": {"choice": "Complaint"}}),
    )
    sample = router.evaluate_sample("it was late")
    assert json.loads(sample.answer) == {"schema": "Complaint", "data": {"topic": "late"}}


def test_an_extra_list_item_is_not_an_exact_match():
    golden = {"lines": [{"item": "tea"}]}
    scores = StructuredMatchMetric().evaluate(_sample({"lines": [{"item": "tea"}, {"item": "cake"}]}, golden))
    assert scores == {"field_accuracy": 1.0, "exact_match": 0.0}
    assert StructuredMatchMetric().evaluate(_sample(golden, golden))["exact_match"] == 1.0
