"""Tests for rag/query_rewriter.py's QueryRewriter (D48)."""

from __future__ import annotations

import time

from kuhaku.core.llm.base import LLMError
from kuhaku.tools.rag import rewrite_cache
from kuhaku.tools.rag.query_rewriter import QueryRewriter


class _ScriptedLLM:
    name = "scripted"

    def __init__(self, response: str | None = None, *, raises: Exception | None = None,
                 sleep_seconds: float = 0.0) -> None:
        self._response = response
        self._raises = raises
        self._sleep_seconds = sleep_seconds
        self.call_count = 0
        self.last_user: str | None = None

    def generate(self, system: str, user: str) -> str:
        self.call_count += 1
        self.last_user = user
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        if self._raises is not None:
            raise self._raises
        assert self._response is not None
        return self._response


def test_rewrite_returns_the_llm_response():
    llm = _ScriptedLLM("central bank policy")
    rewriter = QueryRewriter(llm, timeout_seconds=2.0)
    assert rewriter.rewrite("CB policy") == "central bank policy"
    assert llm.call_count == 1


def test_rewrite_strips_whitespace_from_the_response():
    llm = _ScriptedLLM("  rewritten query text  \n")
    rewriter = QueryRewriter(llm)
    assert rewriter.rewrite("query text") == "rewritten query text"


def test_rewrite_falls_back_to_original_on_llm_error():
    llm = _ScriptedLLM(raises=LLMError("provider unreachable"))
    rewriter = QueryRewriter(llm)
    assert rewriter.rewrite("instant payment limit") == "instant payment limit"


def test_rewrite_falls_back_to_original_on_timeout():
    # A tiny timeout so the test doesn't actually wait ~2s for the default.
    llm = _ScriptedLLM("ignored", sleep_seconds=0.3)
    rewriter = QueryRewriter(llm, timeout_seconds=0.05)
    assert rewriter.rewrite("query text") == "query text"


def test_rewrite_uses_cache_on_second_call(tmp_path):
    db_path = str(tmp_path / "assistant.db")
    llm = _ScriptedLLM("central bank policy")
    rewriter = QueryRewriter(llm, db_path=db_path, cache_ttl_seconds=3600)

    first = rewriter.rewrite("CB policy")
    second = rewriter.rewrite("CB policy")

    assert first == second == "central bank policy"
    assert llm.call_count == 1  # second call was a cache hit


def test_rewrite_populates_the_cache_for_a_pre_seeded_lookup(tmp_path):
    db_path = str(tmp_path / "assistant.db")
    llm = _ScriptedLLM("central bank policy")
    rewriter = QueryRewriter(llm, db_path=db_path, cache_ttl_seconds=3600)
    rewriter.rewrite("CB policy")

    import hashlib

    query_hash = hashlib.sha256("CB policy".encode()).hexdigest()
    assert rewrite_cache.get_rewrite(db_path, query_hash) == "central bank policy"


def test_rewrite_without_db_path_skips_caching_entirely(tmp_path):
    llm = _ScriptedLLM("rewritten")
    rewriter = QueryRewriter(llm, db_path=None)

    rewriter.rewrite("query")
    rewriter.rewrite("query")

    assert llm.call_count == 2  # no cache -> LLM called every time


def test_rewrite_does_not_cache_a_failed_call(tmp_path):
    """A timed-out/failed rewrite must not poison the cache with the original query."""

    db_path = str(tmp_path / "assistant.db")
    llm = _ScriptedLLM(raises=LLMError("boom"))
    rewriter = QueryRewriter(llm, db_path=db_path)
    rewriter.rewrite("query")

    import hashlib

    query_hash = hashlib.sha256(b"query").hexdigest()
    assert rewrite_cache.get_rewrite(db_path, query_hash) is None
