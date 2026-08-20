"""Tests for RAGTargetAdapter's RAG-flavored llm_provider answer-generation fallback --
the piece that used to live inline in the (now tool-agnostic) TargetAdapter."""

from __future__ import annotations

from kuhaku.tools.rag.evaluation_target import RAGTargetAdapter
from kuhaku.tools.rag.models import Chunk, RetrievedChunk


def _rc(document_id: str, text: str = "chunk text") -> RetrievedChunk:
    chunk = Chunk(
        id=f"{document_id}-0",
        document_id=document_id,
        title="title",
        doc_type="doc",
        text=text,
        chunk_index=0,
        source_path="path",
    )
    return RetrievedChunk(chunk=chunk, score=1.0)


class FakeRetriever:
    def __init__(self, rankings: dict[str, list]) -> None:
        self._rankings = rankings

    def retrieve(self, query: str, top_k: int) -> list:
        return self._rankings.get(query, [])[:top_k]


class FakeAskTarget:
    def __init__(self, response) -> None:
        self._response = response

    def ask(self, question: str):
        return self._response


class FakeAgentAnswer:
    def __init__(self, text, retrieved) -> None:
        self.text = text
        self.retrieved = retrieved


class FakeLLMProvider:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._text


def test_retrieve_target_uses_llm_provider_fallback():
    retriever = FakeRetriever({"Q?": [_rc("a")]})
    llm = FakeLLMProvider("llm generated answer")
    adapter = RAGTargetAdapter(retriever, top_k=5, llm_provider=llm)

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer == "llm generated answer"
    assert len(llm.calls) == 1


def test_answer_generator_preferred_over_llm_provider():
    retriever = FakeRetriever({"Q?": [_rc("a")]})
    llm = FakeLLMProvider("should not be used")
    adapter = RAGTargetAdapter(
        retriever,
        top_k=5,
        answer_generator=lambda query, retrieved: "generator answer",
        llm_provider=llm,
    )

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer == "generator answer"
    assert llm.calls == []


def test_ask_target_never_falls_back_to_llm_provider():
    class NoTextAnswer:
        retrieved: list = []

    target = FakeAskTarget(NoTextAnswer())
    llm = FakeLLMProvider("should not be called")
    adapter = RAGTargetAdapter(target, llm_provider=llm)

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer is None
    assert llm.calls == []


def test_ask_target_own_answer_wins_over_llm_provider():
    target = FakeAskTarget(FakeAgentAnswer("engine answer", [_rc("a")]))
    llm = FakeLLMProvider("should not be used")
    adapter = RAGTargetAdapter(target, llm_provider=llm)

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer == "engine answer"
    assert llm.calls == []


def test_no_llm_provider_leaves_answer_none():
    retriever = FakeRetriever({"Q?": [_rc("a")]})
    adapter = RAGTargetAdapter(retriever, top_k=5)

    sample = adapter.evaluate_sample("Q?")

    assert sample.answer is None
