from __future__ import annotations

from dataclasses import dataclass

import pytest

from kuhaku.evaluation.sample import EvaluationSample
from kuhaku.evaluation.target import EvaluationTarget, TargetAdapter


# Generic, tool-agnostic stand-ins for "a retrieved item with a chunk" -- deliberately
# not RAG's own Chunk/RetrievedChunk, since this test file exercises the generic
# TargetAdapter, which must work for any tool's shape, not just RAG's.
@dataclass
class _FakeChunk:
    document_id: str
    text: str


@dataclass
class _FakeRetrievedItem:
    chunk: _FakeChunk


def _item(document_id: str, text: str = "chunk text") -> _FakeRetrievedItem:
    return _FakeRetrievedItem(chunk=_FakeChunk(document_id=document_id, text=text))


class FakeRetriever:
    def __init__(self, rankings: dict[str, list]) -> None:
        self._rankings = rankings
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list:
        self.calls.append((query, top_k))
        return self._rankings.get(query, [])[:top_k]


class RaisingRetriever:
    def retrieve(self, query: str, top_k: int) -> list:
        raise RuntimeError("retriever boom")


class FakeAgentAnswer:
    def __init__(self, text, retrieved) -> None:
        self.text = text
        self.retrieved = retrieved


class FakeAskTarget:
    def __init__(self, response) -> None:
        self._response = response
        self.calls: list[str] = []

    def ask(self, question: str):
        self.calls.append(question)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class NativeTarget:
    """Already implements EvaluationTarget -- TargetAdapter should use it unwrapped."""

    def evaluate_sample(self, query: str) -> EvaluationSample:
        return EvaluationSample(query=query, answer="native answer")


# -- resolution / protocol conformance ------------------------------------------------


def test_native_target_satisfies_protocol_without_wrapping():
    target = NativeTarget()
    assert isinstance(target, EvaluationTarget)


def test_adapter_uses_native_evaluate_sample_directly():
    adapter = TargetAdapter(NativeTarget())
    sample = adapter.evaluate_sample("Q?")
    assert sample.answer == "native answer"


def test_adapter_rejects_object_with_no_recognizable_shape():
    with pytest.raises(TypeError, match="evaluate_sample"):
        TargetAdapter(object())


# -- callable targets -------------------------------------------------------------------


def test_adapter_wraps_callable_returning_string():
    adapter = TargetAdapter(lambda query: f"answer to {query}")
    sample = adapter.evaluate_sample("Q?")
    assert sample.query == "Q?"
    assert sample.answer == "answer to Q?"
    assert sample.contexts is None


def test_adapter_wraps_callable_returning_sample_unchanged():
    def target(query: str) -> EvaluationSample:
        return EvaluationSample(query=query, answer="a", contexts=["c"])

    adapter = TargetAdapter(target)
    sample = adapter.evaluate_sample("Q?")
    assert sample.contexts == ["c"]


def test_adapter_wraps_callable_returning_none():
    adapter = TargetAdapter(lambda query: None)
    sample = adapter.evaluate_sample("Q?")
    assert sample.answer is None


# -- ask()/answer()-style targets --------------------------------------------------------


def test_adapter_ask_target_extracts_answer_and_contexts():
    target = FakeAskTarget(FakeAgentAnswer("generated answer", [_item("a"), _item("b")]))
    adapter = TargetAdapter(target)

    sample = adapter.evaluate_sample("Q?")

    assert target.calls == ["Q?"]
    assert sample.answer == "generated answer"
    assert sample.contexts == ["chunk text", "chunk text"]
    assert sample.retrieved_doc_ids == ["a", "b"]


def test_adapter_ask_target_failure_returns_empty_sample():
    target = FakeAskTarget(RuntimeError("engine boom"))
    adapter = TargetAdapter(target)

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer is None
    assert sample.contexts is None


def test_adapter_answer_generator_overrides_ask_target_answer():
    target = FakeAskTarget(FakeAgentAnswer("should not be used", [_item("a")]))
    adapter = TargetAdapter(
        target, answer_generator=lambda query, retrieved: "explicit generator answer"
    )

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer == "explicit generator answer"


def test_adapter_ask_target_with_no_text_leaves_answer_none():
    class NoTextAnswer:
        retrieved: list = []

    target = FakeAskTarget(NoTextAnswer())
    adapter = TargetAdapter(target)

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer is None


# -- retrieve()/search()-style targets ----------------------------------------------------


def test_adapter_retrieve_target_extracts_contexts_and_ids():
    retriever = FakeRetriever({"Q?": [_item("a"), _item("b")]})
    adapter = TargetAdapter(retriever, top_k=2)

    sample = adapter.evaluate_sample("Q?")

    assert retriever.calls == [("Q?", 2)]
    assert sample.contexts == ["chunk text", "chunk text"]
    assert sample.retrieved_doc_ids == ["a", "b"]
    assert sample.answer is None


def test_adapter_retrieve_target_accepts_bare_string_ids():
    retriever = FakeRetriever({"Q?": ["a", "b"]})
    adapter = TargetAdapter(retriever, top_k=2)

    sample = adapter.evaluate_sample("Q?")

    assert sample.retrieved_doc_ids == ["a", "b"]


def test_adapter_retrieve_failure_returns_empty_contexts():
    adapter = TargetAdapter(RaisingRetriever(), top_k=5)

    sample = adapter.evaluate_sample("Q?")

    assert sample.contexts == []
    assert sample.retrieved_doc_ids == []


def test_adapter_retrieve_target_answer_generator_used():
    retriever = FakeRetriever({"Q?": [_item("a")]})
    adapter = TargetAdapter(
        retriever, top_k=5, answer_generator=lambda query, retrieved: "generator answer"
    )

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer == "generator answer"


def test_adapter_retrieve_target_no_answer_source_leaves_answer_none():
    retriever = FakeRetriever({"Q?": ["a"]})
    adapter = TargetAdapter(retriever, top_k=5)

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer is None
