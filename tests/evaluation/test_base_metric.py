from __future__ import annotations

import pytest

from kuhaku.evaluation.base_metric import BaseMetric
from kuhaku.evaluation.sample import EvaluationSample


def test_base_metric_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseMetric()


def test_base_metric_subclass_missing_evaluate_cannot_be_instantiated():
    class Incomplete(BaseMetric):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_base_metric_subclass_can_be_instantiated_and_used():
    class AlwaysOne(BaseMetric):
        name = "always_one"

        def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
            return {"always_one": 1.0}

    metric = AlwaysOne()
    assert metric.name == "always_one"
    assert metric.evaluate(EvaluationSample(query="Q?")) == {"always_one": 1.0}
