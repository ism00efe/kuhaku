from __future__ import annotations

import json

import pytest

from kuhaku.tools.rag.models import Chunk, RetrievedChunk
from kuhaku.evaluation.metrics import (
    FaithfulnessMetric,
    HitRateAtKMetric,
    MRRMetric,
    RecallAtKMetric,
)
from kuhaku.evaluation.runner import EvaluationRunner
from kuhaku.evaluation.sample import EvaluationSample
from kuhaku.evaluation.store import InMemoryEvaluationStore


def _chunk(document_id: str) -> Chunk:
    return Chunk(
        id=f"{document_id}-0",
        document_id=document_id,
        title="title",
        doc_type="doc",
        text=f"text for {document_id}",
        chunk_index=0,
        source_path="path",
    )


def _rc(document_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(document_id), score=1.0)


def _write_dataset(tmp_path, items: list[dict]):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
    return path


class FakeRetriever:
    """Returns a deterministic, pre-wired ranking of document ids per question."""

    def __init__(self, rankings: dict[str, list]) -> None:
        self._rankings = rankings
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list:
        self.calls.append((query, top_k))
        return self._rankings.get(query, [])[:top_k]


class RaisingRetriever:
    def retrieve(self, query: str, top_k: int) -> list:
        raise RuntimeError("retriever boom")


class FakeRAGAnswer:
    def __init__(self, text: str, retrieved: list[RetrievedChunk]) -> None:
        self.text = text
        self.retrieved = retrieved


class FakeRAGTarget:
    def __init__(self, response) -> None:
        self._response = response
        self.calls: list[str] = []

    def ask(self, question: str):
        self.calls.append(question)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class NativeTarget:
    """A non-RAG target: produces EvaluationSample directly, no retrieve/ask at all."""

    def __init__(self, samples: dict[str, EvaluationSample]) -> None:
        self._samples = samples
        self.calls: list[str] = []

    def evaluate_sample(self, query: str) -> EvaluationSample:
        self.calls.append(query)
        return self._samples[query]


class RaisingNativeTarget:
    def evaluate_sample(self, query: str) -> EvaluationSample:
        raise RuntimeError("target boom")


class FakeLLMProvider:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._text


class RaisingMetric:
    name = "raising"

    def evaluate(self, sample: EvaluationSample):
        raise RuntimeError("metric boom")


# -- basic orchestration over a retriever target (via TargetAdapter) -----------------


def test_run_retriever_target_computes_metrics(tmp_path):
    dataset_path = _write_dataset(
        tmp_path,
        [
            {"question_id": "q1", "question": "Q1?", "expected_sources": ["a"]},
            {"question_id": "q2", "question": "Q2?", "expected_sources": ["z"]},
        ],
    )
    retriever = FakeRetriever({"Q1?": ["a", "b"], "Q2?": ["x", "y"]})
    runner = EvaluationRunner([RecallAtKMetric(k=2), MRRMetric(), HitRateAtKMetric(k=2)], dataset_path)

    summary = runner.run(retriever, top_k=2)

    assert retriever.calls == [("Q1?", 2), ("Q2?", 2)]
    assert summary["hit_rate_at_k"] == 0.5
    assert summary["mrr"] == 0.5
    assert set(summary) == {"recall_at_k", "mrr", "hit_rate_at_k"}


def test_run_defaults_to_in_memory_store(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    retriever = FakeRetriever({"Q?": ["a"]})
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path)

    summary = runner.run(retriever, top_k=5)

    assert summary["recall_at_k"] == 1.0


def test_run_uses_provided_store(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    retriever = FakeRetriever({"Q?": ["a"]})
    store = InMemoryEvaluationStore()
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path, store=store)

    runner.run(retriever, top_k=5)

    assert len(store.results) == 1
    assert store.results[0]["question_id"] == "q1"
    assert store.results[0]["retrieved_ids"] == ["a"]


def test_run_no_chunks_retrieved_scores_all_zero(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    retriever = FakeRetriever({})
    runner = EvaluationRunner([RecallAtKMetric(k=5), MRRMetric()], dataset_path)

    summary = runner.run(retriever, top_k=5)

    assert all(value == 0.0 for value in summary.values())


def test_run_empty_dataset_returns_empty_dict(tmp_path):
    dataset_path = _write_dataset(tmp_path, [])
    retriever = FakeRetriever({})
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path)

    assert runner.run(retriever, top_k=5) == {}


# -- target adaptation -----------------------------------------------------------------


def test_run_target_without_recognizable_shape_raises_type_error(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path)

    with pytest.raises(TypeError, match="evaluate_sample"):
        runner.run(object())


def test_run_native_evaluation_target_used_directly_no_adaptation(tmp_path):
    """A tool-agnostic target -- no retrieve/ask at all -- is scored the same way."""

    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    target = NativeTarget(
        {"Q?": EvaluationSample(query="Q?", answer="native answer", retrieved_doc_ids=["a"])}
    )
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path)

    summary = runner.run(target)

    assert target.calls == ["Q?"]
    assert summary["recall_at_k"] == 1.0


# -- RAG target: retrieved + answer come from a single ask() call ---------------------


def test_run_rag_target_reuses_single_ask_call_for_retrieved_and_answer(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    target = FakeRAGTarget(FakeRAGAnswer("generated answer", [_rc("a")]))
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path)

    summary = runner.run(target)

    assert target.calls == ["Q?"]
    assert summary["recall_at_k"] == 1.0


def test_run_answer_generator_preferred_over_rag_target(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    target = FakeRAGTarget(FakeRAGAnswer("should not be used", [_rc("a")]))
    store = InMemoryEvaluationStore()
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path, store=store)

    runner.run(
        target,
        answer_generator=lambda question, retrieved: "explicit generator answer",
    )

    assert store.results[0]["answer"] == "explicit generator answer"


def test_run_rag_target_adapter_llm_provider_fallback_used_for_retriever_target(tmp_path):
    """The generic runner has no notion of `llm_provider` -- a caller who wants that
    RAG-flavored fallback constructs a `RAGTargetAdapter` explicitly and passes it as
    the target, already-adapted (see kuhaku.tools.rag.evaluation_target)."""

    from kuhaku.tools.rag.evaluation_target import RAGTargetAdapter

    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    retriever = FakeRetriever({"Q?": [_rc("a")]})
    llm = FakeLLMProvider("llm generated answer")
    store = InMemoryEvaluationStore()
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path, store=store)

    runner.run(RAGTargetAdapter(retriever, top_k=5, llm_provider=llm))

    assert store.results[0]["answer"] == "llm generated answer"
    assert len(llm.calls) == 1


def test_run_no_answer_source_leaves_answer_none(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    retriever = FakeRetriever({"Q?": ["a"]})
    store = InMemoryEvaluationStore()
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path, store=store)

    runner.run(retriever, top_k=5)

    assert store.results[0]["answer"] is None


# -- error handling: a single bad question/metric/target call never crashes the run ---


def test_run_retrieve_failure_continues_with_empty_retrieved(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path)

    summary = runner.run(RaisingRetriever(), top_k=5)

    assert summary["recall_at_k"] == 0.0


def test_run_ask_failure_continues_with_no_answer(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    target = FakeRAGTarget(RuntimeError("engine boom"))
    store = InMemoryEvaluationStore()
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path, store=store)

    summary = runner.run(target)

    assert summary["recall_at_k"] == 0.0
    assert store.results[0]["answer"] is None


def test_run_native_target_failure_continues_with_empty_sample(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path)

    summary = runner.run(RaisingNativeTarget())

    assert summary["recall_at_k"] == 0.0


def test_run_metric_failure_is_skipped_and_run_continues(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    retriever = FakeRetriever({"Q?": ["a"]})
    runner = EvaluationRunner([RaisingMetric(), RecallAtKMetric(k=5)], dataset_path)

    summary = runner.run(retriever, top_k=5)

    assert summary == {"recall_at_k": 1.0}


def test_run_judge_failure_via_faithfulness_metric_continues(tmp_path):
    class FailingJudge:
        def evaluate(self, question, context_chunks, generated_answer):
            raise RuntimeError("judge boom")

    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    target = FakeRAGTarget(FakeRAGAnswer("an answer", [_rc("a")]))
    runner = EvaluationRunner(
        [RecallAtKMetric(k=5), FaithfulnessMetric(judge=FailingJudge())], dataset_path
    )

    summary = runner.run(target)

    assert "faithfulness" not in summary
    assert summary["recall_at_k"] == 1.0


# -- run_id ----------------------------------------------------------------------------


def test_run_generates_unique_run_id_when_not_provided(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    store = InMemoryEvaluationStore()
    EvaluationRunner([RecallAtKMetric(k=5)], dataset_path, store=store).run(FakeRetriever({"Q?": ["a"]}))
    EvaluationRunner([RecallAtKMetric(k=5)], dataset_path, store=store).run(FakeRetriever({"Q?": ["a"]}))

    run_ids = {r["run_id"] for r in store.results}
    assert len(run_ids) == 2


def test_run_explicit_run_id_isolates_results_in_shared_store(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    store = InMemoryEvaluationStore()
    EvaluationRunner(
        [RecallAtKMetric(k=5)], dataset_path, store=store, run_id="run-a"
    ).run(FakeRetriever({"Q?": ["a"]}))
    EvaluationRunner(
        [RecallAtKMetric(k=5)], dataset_path, store=store, run_id="run-b"
    ).run(FakeRetriever({}))

    assert store.summary(run_id="run-a") == {"recall_at_k": 1.0}
    assert store.summary(run_id="run-b") == {"recall_at_k": 0.0}


# -- golden merge ------------------------------------------------------------------------


def test_run_merges_golden_answer_into_sample_metadata_for_answer_correctness(tmp_path):
    from kuhaku.evaluation.metrics import AnswerCorrectnessMetric

    dataset_path = _write_dataset(
        tmp_path,
        [
            {
                "question_id": "q1",
                "question": "Q?",
                "expected_sources": ["a"],
                "golden_answer": "a b c",
            }
        ],
    )
    target = FakeRAGTarget(FakeRAGAnswer("a b d", [_rc("a")]))
    runner = EvaluationRunner([AnswerCorrectnessMetric()], dataset_path)

    summary = runner.run(target)

    assert summary["answer_correctness"] == pytest.approx(2 / 4)


def test_run_does_not_overwrite_target_supplied_relevant_doc_ids(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["golden-doc"]}]
    )
    target = NativeTarget(
        {"Q?": EvaluationSample(query="Q?", retrieved_doc_ids=["self-doc"], relevant_doc_ids={"self-doc"})}
    )
    runner = EvaluationRunner([RecallAtKMetric(k=5)], dataset_path)

    summary = runner.run(target)

    assert summary["recall_at_k"] == 1.0


# -- judge auto-creation (Feature 5) ----------------------------------------------------


def test_runner_uses_explicit_judge_llm_provider(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    target = FakeRAGTarget(FakeRAGAnswer("an answer", [_rc("a")]))
    judge_llm = FakeLLMProvider("0.9")
    metric = FaithfulnessMetric()
    runner = EvaluationRunner([metric], dataset_path, judge_llm_provider=judge_llm)

    summary = runner.run(target)

    assert summary["faithfulness"] == 0.9
    assert len(judge_llm.calls) == 1


def test_runner_auto_creates_zero_temperature_judge_from_settings(tmp_path):
    from kuhaku.core.config import Settings

    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    target = FakeRAGTarget(FakeRAGAnswer("an answer", [_rc("a")]))
    metric = FaithfulnessMetric()
    settings = Settings(llm_provider="ollama", llm_temperature=0.7)
    runner = EvaluationRunner([metric], dataset_path, settings=settings)

    assert metric.judge is None  # not wired until run() so a metric can still opt out
    runner.run(target)

    assert metric.judge is not None


def test_runner_without_judge_provider_or_settings_leaves_faithfulness_unset(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    target = FakeRAGTarget(FakeRAGAnswer("an answer", [_rc("a")]))
    metric = FaithfulnessMetric()
    runner = EvaluationRunner([metric], dataset_path)

    summary = runner.run(target)

    assert metric.judge is None
    assert "faithfulness" not in summary


def test_runner_explicit_metric_judge_injection_preserved_over_auto_creation(tmp_path):
    dataset_path = _write_dataset(
        tmp_path, [{"question_id": "q1", "question": "Q?", "expected_sources": ["a"]}]
    )
    target = FakeRAGTarget(FakeRAGAnswer("an answer", [_rc("a")]))

    class StubJudge:
        def evaluate(self, question, context_chunks, generated_answer) -> float:
            return 0.42

    explicit_metric = FaithfulnessMetric(judge=StubJudge())
    auto_judge_llm = FakeLLMProvider("0.9")
    runner = EvaluationRunner(
        [explicit_metric], dataset_path, judge_llm_provider=auto_judge_llm
    )

    summary = runner.run(target)

    assert summary["faithfulness"] == 0.42
    assert auto_judge_llm.calls == []
