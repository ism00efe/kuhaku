"""LLM-based, togglable pre-retrieval query rewriting.

``QueryRewriter`` is the single orchestration point for "hash the query -> check the
SQLite rewrite cache -> call the LLM under a timeout -> cache the result -> log/record
a metric". It is deliberately the only place this sequence is implemented: both the
live pipeline (``rag/engine.py``'s ``RAGEngine``) and the evaluation harness
(``evaluation/runner.py``'s ``BenchmarkRunner``) construct one instance and call
``.rewrite()`` -- neither duplicates the cache-lookup or timeout logic.

Failure is always graceful: a timed-out or failed LLM call returns the original query
unchanged rather than raising, so the pipeline never fails because rewriting failed.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

from kuhaku.core.llm.base import LLMError, LLMProvider

from . import rewrite_cache
from .metrics import record_query_rewrite

logger = logging.getLogger(__name__)

# Domain constants externalization: the prompt text itself lives in
# ``prompts/query_rewriter_system_prompt.txt`` (this package's own data directory),
# mirroring ``rag/prompts.py``'s ``SYSTEM_PROMPT_PATH`` convention -- resolved relative to
# this module's file, not the process's CWD, so `kuhaku` works the same whether run
# from source in this repo or `pip install`ed standalone. Unlike that file, a missing
# prompt file here degrades to this built-in default rather than failing at import time --
# see ``_load_default_system_prompt``.
_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "query_rewriter_system_prompt.txt"
)

# Last-resort fallback, byte-identical to the prompt file's contents, used only if the
# file is missing (e.g. an unusual working directory). Rewrites are retrieval-only (see
# RAGEngine integration): the LLM-facing prompt and the guard still see
# the user's original text, so a rewrite that drifts from the original intent only ever
# costs recall, never changes what the user is told or what security inspects.
_DEFAULT_SYSTEM_PROMPT = """You are a query optimizer for a document retrieval system.
Rewrite the user's query to improve retrieval performance. Rules:
1. Expand abbreviations and acronyms where the expansion is unambiguous
2. Use formal, precise terminology where applicable
3. Fix typos and spelling errors
4. Preserve the original intent — do not change the meaning
5. Return ONLY the rewritten query, no explanations, no markdown
6. If the query is already well-formed, return it as-is"""


def _load_default_system_prompt() -> str:
    try:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning(
            "query rewriter system prompt file '%s' not found; using built-in default",
            _SYSTEM_PROMPT_PATH,
        )
        return _DEFAULT_SYSTEM_PROMPT


class QueryRewriter:
    """Rewrites a retrieval query via an LLM, with caching and graceful degradation."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        db_path: str | None = None,
        cache_ttl_seconds: int = 3600,
        timeout_seconds: float = 2.0,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._db_path = db_path
        self._cache_ttl_seconds = cache_ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._system_prompt = (
            system_prompt if system_prompt is not None else _load_default_system_prompt()
        )

    def rewrite(self, query: str) -> str:
        """Return a retrieval-optimized rewrite of ``query``, or ``query`` unchanged."""

        start = time.monotonic()
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()

        if self._db_path is not None:
            cached = rewrite_cache.get_rewrite(self._db_path, query_hash)
            if cached is not None:
                self._log(query, cached, cache_hit=True, success=True, start=start)
                return cached

        rewritten, success = self._call_llm(query)
        if success and self._db_path is not None:
            rewrite_cache.save_rewrite(
                self._db_path, query_hash, rewritten, self._cache_ttl_seconds
            )
        self._log(query, rewritten, cache_hit=False, success=success, start=start)
        return rewritten

    def _call_llm(self, query: str) -> tuple[str, bool]:
        # LLMProvider.generate() has no per-call timeout (temperature/timeout are
        # constructor-only). signal.alarm isn't available on Windows, so a short-lived
        # single-worker executor is the portable way to bound this call.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._llm.generate, self._system_prompt, query)
            try:
                result = future.result(timeout=self._timeout_seconds)
                return result.strip() or query, True
            except FutureTimeoutError:
                logger.warning(
                    "query rewrite timed out after %.1fs; using original query",
                    self._timeout_seconds,
                )
                return query, False
            except LLMError:
                logger.warning(
                    "query rewrite LLM call failed; using original query", exc_info=True
                )
                return query, False

    def _log(
        self, original: str, rewritten: str, *, cache_hit: bool, success: bool, start: float
    ) -> None:
        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "query rewrite",
            extra={
                "event": "query_rewrite",
                "original": original,
                "rewritten": rewritten,
                "cache_hit": cache_hit,
                "duration_ms": round(duration_ms, 2),
            },
        )
        record_query_rewrite(
            cache_hit="true" if cache_hit else "false",
            success="true" if success else "false",
        )
