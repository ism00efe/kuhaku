"""How classification sits in kuhaku: tracing, evaluation, the public surface, the
dependency rule."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

import kuhaku
from kuhaku.core.observability import bind_trace_id
from kuhaku.evaluation import (
    AnswerCorrectnessMetric,
    EvaluationRunner,
    EvaluationTarget,
)
from kuhaku.tools.classification import Classifier, KeywordRule

from .conftest import LABELS, ScriptedLLM, answer

SRC = Path(kuhaku.__file__).parent


def _imports(package: Path) -> set[str]:
    found: set[str] = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
    return found


def test_classification_depends_on_core_but_never_on_another_tool():
    modules = _imports(SRC / "tools" / "classification")
    assert any(module.startswith("kuhaku.core") for module in modules)
    assert not any(module.startswith("kuhaku.tools.") for module in modules)


def test_core_never_depends_on_classification():
    assert not any(
        module.startswith("kuhaku.tools.classification") for module in _imports(SRC / "core")
    )


def test_entry_points_are_exported_from_kuhaku():
    for name in (
        "Classifier",
        "Label",
        "ClassificationResult",
        "ClassificationStatus",
        "UndecidedReason",
    ):
        assert name in kuhaku.__all__
    assert kuhaku.Classifier is Classifier


def test_every_stage_is_logged_under_the_results_trace_id(caplog):
    clf = Classifier(
        LABELS, llm=ScriptedLLM(answer(("billing", 0.9))), rules=[KeywordRule("account", ["x"])]
    )
    with caplog.at_level(logging.INFO, logger="kuhaku.tools.classification"):
        result = clf.classify("refund", trace_id="t-1")

    assert result.trace_id == "t-1"
    steps = {getattr(record, "step", None) for record in caplog.records}
    assert {"classification", "classification.rules", "classification.classify"} <= steps
    summary = next(r for r in caplog.records if getattr(r, "step", None) == "classification")
    assert summary.outcome == "decided"
    assert summary.decided_by == "classify"
    assert all(getattr(r, "trace_id", "t-1") == "t-1" for r in caplog.records)


def test_an_outer_trace_id_is_inherited():
    clf = Classifier(LABELS, llm=ScriptedLLM(answer(("billing", 0.9))))
    with bind_trace_id("outer"):
        assert clf.classify("refund").trace_id == "outer"


def test_a_classifier_is_an_evaluation_target():
    clf = Classifier(
        LABELS, llm=ScriptedLLM(answer(("billing", 0.9)), {"labels": []}), multi_label=True
    )
    assert isinstance(clf, EvaluationTarget)

    decided = clf.evaluate_sample("refund")
    assert decided.answer == "billing"
    assert decided.metadata["labels"] == ["billing"]
    assert decided.metadata["decided_by"] == "classify"

    undecided = clf.evaluate_sample("recipe")
    assert undecided.answer is None
    assert undecided.metadata["undecided"] == "no_label"


def test_answer_correctness_scores_label_accuracy(tmp_path):
    rows = [
        {"question_id": "q1", "question": "refund please", "golden_answer": "billing"},
        {"question_id": "q2", "question": "app crashes", "golden_answer": "technical"},
    ]
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    llm = ScriptedLLM(answer(("billing", 0.9)), answer(("account", 0.9)))

    runner = EvaluationRunner([AnswerCorrectnessMetric()], dataset)
    summary = runner.run(Classifier(LABELS, llm=llm))

    assert summary["answer_correctness"] == 0.5
