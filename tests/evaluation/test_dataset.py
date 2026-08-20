from __future__ import annotations

import json

import pytest

from kuhaku.evaluation.dataset import EvaluationItem, load_eval_dataset


def _write(tmp_path, lines: list[str]):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_loads_full_item(tmp_path):
    line = json.dumps(
        {
            "question_id": "q1",
            "question": "What is RC-05?",
            "expected_sources": ["doc_a"],
            "golden_chunks": ["doc_a::0"],
            "golden_answer": "RC-05 means declined.",
            "category": "lexical",
        }
    )
    items = load_eval_dataset(_write(tmp_path, [line]))
    assert items == [
        EvaluationItem(
            question_id="q1",
            question="What is RC-05?",
            expected_sources=["doc_a"],
            golden_chunks=["doc_a::0"],
            golden_answer="RC-05 means declined.",
            category="lexical",
        )
    ]


def test_empty_file_returns_empty_list(tmp_path):
    assert load_eval_dataset(_write(tmp_path, [])) == []


def test_blank_lines_are_skipped(tmp_path):
    line = json.dumps({"question_id": "q1", "question": "Q?"})
    items = load_eval_dataset(_write(tmp_path, ["", line, "   "]))
    assert len(items) == 1


def test_missing_expected_sources_defaults_to_empty_list(tmp_path):
    line = json.dumps({"question_id": "q1", "question": "Q?"})
    items = load_eval_dataset(_write(tmp_path, [line]))
    assert items[0].expected_sources == []
    assert items[0].golden_chunks == []
    assert items[0].golden_answer == ""
    assert items[0].category is None


def test_missing_required_field_raises_value_error(tmp_path):
    line = json.dumps({"question_id": "q1"})
    with pytest.raises(ValueError, match="question"):
        load_eval_dataset(_write(tmp_path, [line]))


def test_malformed_json_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="malformed JSON"):
        load_eval_dataset(_write(tmp_path, ["{not valid json"]))


def test_wrong_type_raises_value_error(tmp_path):
    line = json.dumps({"question_id": "q1", "question": "Q?", "expected_sources": "not_a_list"})
    with pytest.raises(ValueError, match="expected_sources"):
        load_eval_dataset(_write(tmp_path, [line]))


def test_wrong_type_for_required_field_raises_value_error(tmp_path):
    line = json.dumps({"question_id": 123, "question": "Q?"})
    with pytest.raises(ValueError, match="question_id"):
        load_eval_dataset(_write(tmp_path, [line]))
