"""Integration tests: RAGEngine + QueryRewriter wired together (D48).

Verifies the scoping decision from DECISIONS.md D48: rewriting must affect only the
string handed to the retriever -- not the guard, not the QA-cache key, not the
generation prompt (all of which keep seeing the original `retrieval_query`).
"""

from __future__ import annotations

from kuhaku.tools.rag.engine import RAGEngine
from kuhaku.tools.rag.query_rewriter import QueryRewriter
from tests.conftest import FakeEmbeddings, FakeLLM, FakeRetriever, FakeVectorStore, make_chunk


class _ScriptedRewriteLLM:
    name = "scripted-rewriter"

    def __init__(self, rewritten: str) -> None:
        self._rewritten = rewritten
        self.call_count = 0

    def generate(self, system: str, user: str) -> str:
        self.call_count += 1
        return self._rewritten


def test_rewriting_enabled_sends_the_rewritten_query_to_the_retriever():
    retriever = FakeRetriever([make_chunk("faq_general")])
    rewrite_llm = _ScriptedRewriteLLM("central bank limit")
    rewriter = QueryRewriter(rewrite_llm)
    engine = RAGEngine(
        FakeEmbeddings(),
        FakeVectorStore([make_chunk("x")]),
        FakeLLM("Answer [S1]."),
        retriever=retriever,
        query_rewriter=rewriter,
    )

    engine.answer("what is the CB limit?", None)

    assert retriever.calls[0][0] == "central bank limit"


def test_rewriting_disabled_sends_the_original_query_to_the_retriever():
    retriever = FakeRetriever([make_chunk("faq_general")])
    engine = RAGEngine(
        FakeEmbeddings(),
        FakeVectorStore([make_chunk("x")]),
        FakeLLM("Answer [S1]."),
        retriever=retriever,
        query_rewriter=None,
    )

    engine.answer("what is the CB limit?", None)

    assert retriever.calls[0][0] == "what is the CB limit?"


def test_rewriting_does_not_change_the_generation_prompt_the_llm_receives():
    """The LLM-facing prompt must reflect the user's original phrasing, not the rewrite
    -- rewriting is a retrieval-only optimization (DECISIONS.md D48)."""

    retriever = FakeRetriever([make_chunk("faq_general", text="context text")])
    rewrite_llm = _ScriptedRewriteLLM("central bank limit")
    rewriter = QueryRewriter(rewrite_llm)
    answer_llm = FakeLLM("Answer [S1].")
    engine = RAGEngine(
        FakeEmbeddings(),
        FakeVectorStore([make_chunk("x")]),
        answer_llm,
        retriever=retriever,
        query_rewriter=rewriter,
    )

    engine.answer("what is the CB limit?", None)

    assert "what is the CB limit?" in answer_llm.last_user
    assert "central bank limit" not in answer_llm.last_user


def test_two_calls_with_caching_enabled_invoke_the_rewrite_llm_once(tmp_path):
    db_path = str(tmp_path / "assistant.db")
    retriever = FakeRetriever([make_chunk("faq_general")])
    rewrite_llm = _ScriptedRewriteLLM("central bank limit")
    rewriter = QueryRewriter(rewrite_llm, db_path=db_path)
    engine = RAGEngine(
        FakeEmbeddings(),
        FakeVectorStore([make_chunk("x")]),
        FakeLLM("Answer [S1]."),
        retriever=retriever,
        query_rewriter=rewriter,
    )

    engine.answer("what is the CB limit?", None)
    engine.answer("what is the CB limit?", None)

    assert rewrite_llm.call_count == 1
    assert retriever.calls[0][0] == retriever.calls[1][0] == "central bank limit"
