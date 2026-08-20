from __future__ import annotations

from kuhaku.evaluation.sample import EvaluationSample


def test_defaults():
    sample = EvaluationSample(query="Q?")
    assert sample.answer is None
    assert sample.contexts is None
    assert sample.relevant_doc_ids is None
    assert sample.retrieved_doc_ids is None
    assert sample.tool_calls is None
    assert sample.metadata == {}


def test_with_golden_sets_relevant_doc_ids_and_metadata():
    sample = EvaluationSample(query="Q?", answer="a")
    merged = sample.with_golden(
        relevant_doc_ids={"doc-1"}, metadata={"golden_answer": "golden", "question_id": "q1"}
    )
    assert merged.relevant_doc_ids == {"doc-1"}
    assert merged.metadata == {"golden_answer": "golden", "question_id": "q1"}
    # the original sample is untouched (frozen, immutable merge)
    assert sample.relevant_doc_ids is None
    assert sample.metadata == {}


def test_with_golden_does_not_overwrite_target_supplied_relevant_doc_ids():
    sample = EvaluationSample(query="Q?", relevant_doc_ids={"self-judged"})
    merged = sample.with_golden(relevant_doc_ids={"golden-doc"}, metadata={})
    assert merged.relevant_doc_ids == {"self-judged"}


def test_with_golden_merges_metadata_with_golden_precedence():
    sample = EvaluationSample(query="Q?", metadata={"trace_id": "t1", "golden_answer": "stale"})
    merged = sample.with_golden(relevant_doc_ids=set(), metadata={"golden_answer": "fresh"})
    assert merged.metadata == {"trace_id": "t1", "golden_answer": "fresh"}
