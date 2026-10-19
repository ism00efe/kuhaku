"""The suite must leave nothing behind in the working tree (AGENTS.md)."""

from __future__ import annotations

from pathlib import Path

from kuhaku.core.security import record_audit
from kuhaku.tools.rag.cache import QueryAnswerCache

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tests_do_not_run_from_the_repository():
    assert _REPO_ROOT not in (Path.cwd().resolve(), *Path.cwd().resolve().parents)


def test_default_audit_and_cache_paths_stay_out_of_the_repository():
    record_audit(
        "", trace_id="t", raw_question="q", sanitized_retrieval_query="q", event_type="ok"
    )
    cache = QueryAnswerCache("", ttl_seconds=60)
    cache.put("k", "v")

    assert (Path.cwd() / "logs" / "kuhaku_audit.jsonl").exists()
    assert (Path.cwd() / "data" / "kuhaku_qa_cache.sqlite3").exists()
    assert not (_REPO_ROOT / "logs").exists()
    assert not (_REPO_ROOT / "data").exists()
