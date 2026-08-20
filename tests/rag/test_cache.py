"""Tests for the SQLite-backed query-answer cache (FR2, Category 2)."""

from __future__ import annotations

import logging

from kuhaku.tools.rag.cache import QueryAnswerCache, compute_cache_key


# --- compute_cache_key ---------------------------------------------------------
def test_compute_cache_key_is_deterministic():
    a = compute_cache_key("soru", ["d1::0", "d2::0"], "v1", "llm-v1", "embed-v1")
    b = compute_cache_key("soru", ["d1::0", "d2::0"], "v1", "llm-v1", "embed-v1")
    assert a == b


def test_compute_cache_key_is_sensitive_to_chunk_order():
    """[S#] tags are positional -- a reordered retrieval must not hit the same key."""

    a = compute_cache_key("soru", ["d1::0", "d2::0"], "v1", "llm-v1", "embed-v1")
    b = compute_cache_key("soru", ["d2::0", "d1::0"], "v1", "llm-v1", "embed-v1")
    assert a != b


def test_compute_cache_key_is_sensitive_to_query():
    a = compute_cache_key("soru bir", ["d1::0"], "v1", "llm-v1", "embed-v1")
    b = compute_cache_key("soru iki", ["d1::0"], "v1", "llm-v1", "embed-v1")
    assert a != b


def test_compute_cache_key_is_sensitive_to_chunk_set():
    a = compute_cache_key("soru", ["d1::0"], "v1", "llm-v1", "embed-v1")
    b = compute_cache_key("soru", ["d1::0", "d2::0"], "v1", "llm-v1", "embed-v1")
    assert a != b


def test_compute_cache_key_is_sensitive_to_prompt_version():
    a = compute_cache_key("soru", ["d1::0"], "v1", "llm-v1", "embed-v1")
    b = compute_cache_key("soru", ["d1::0"], "v2", "llm-v1", "embed-v1")
    assert a != b


def test_compute_cache_key_is_sensitive_to_llm_version():
    """D42, verification item 1: an LLM upgrade must invalidate every cached answer."""

    a = compute_cache_key("soru", ["d1::0"], "v1", "llm-v1", "embed-v1")
    b = compute_cache_key("soru", ["d1::0"], "v1", "llm-v2", "embed-v1")
    assert a != b


def test_compute_cache_key_is_sensitive_to_embedding_version():
    """D42, verification item 1: an embedding-model upgrade must invalidate the cache."""

    a = compute_cache_key("soru", ["d1::0"], "v1", "llm-v1", "embed-v1")
    b = compute_cache_key("soru", ["d1::0"], "v1", "llm-v1", "embed-v2")
    assert a != b


# --- get/put ---------------------------------------------------------------------
def test_put_then_get_within_ttl_returns_the_cached_text(tmp_path):
    cache = QueryAnswerCache(str(tmp_path / "cache.db"), ttl_seconds=3600)
    cache.put("key1", "Answer [S1].")
    assert cache.get("key1") == "Answer [S1]."


def test_get_is_none_for_a_missing_key(tmp_path):
    cache = QueryAnswerCache(str(tmp_path / "cache.db"), ttl_seconds=3600)
    assert cache.get("missing") is None


def test_cached_entry_becomes_unreachable_after_a_version_bump(tmp_path):
    """D42, verification item 6: an entry written under one deployment's versions must
    miss once any of llm/embedding/prompt version changes -- computing the *new* key and
    looking it up is exactly what RAGEngine does on every request, so this is the
    behavior a real model/prompt upgrade produces."""

    cache = QueryAnswerCache(str(tmp_path / "cache.db"), ttl_seconds=3600)
    old_key = compute_cache_key("soru", ["d1::0"], "v1", "llm-v1", "embed-v1")
    cache.put(old_key, "eski yanıt")
    assert cache.get(old_key) == "eski yanıt"

    new_key = compute_cache_key("soru", ["d1::0"], "v1", "llm-v2", "embed-v1")
    assert cache.get(new_key) is None


def test_put_overwrites_an_existing_key(tmp_path):
    cache = QueryAnswerCache(str(tmp_path / "cache.db"), ttl_seconds=3600)
    cache.put("key1", "first")
    cache.put("key1", "second")
    assert cache.get("key1") == "second"


def test_get_after_ttl_expiry_returns_none(tmp_path, monkeypatch):
    import kuhaku.tools.rag.cache as cache_module

    now = [1_000_000.0]
    monkeypatch.setattr(cache_module.time, "time", lambda: now[0])

    cache = QueryAnswerCache(str(tmp_path / "cache.db"), ttl_seconds=60)
    cache.put("key1", "Answer [S1].")
    assert cache.get("key1") == "Answer [S1]."

    now[0] += 61  # past the 60s TTL
    assert cache.get("key1") is None


def test_get_within_ttl_boundary_still_hits(tmp_path, monkeypatch):
    import kuhaku.tools.rag.cache as cache_module

    now = [1_000_000.0]
    monkeypatch.setattr(cache_module.time, "time", lambda: now[0])

    cache = QueryAnswerCache(str(tmp_path / "cache.db"), ttl_seconds=60)
    cache.put("key1", "Answer [S1].")

    now[0] += 59  # still inside the window
    assert cache.get("key1") == "Answer [S1]."


# --- best-effort degradation ------------------------------------------------------
def test_get_and_put_against_a_broken_path_degrade_gracefully_and_warn(tmp_path, caplog):
    # A directory where the "database file" is expected: sqlite3.connect fails against
    # it on every platform (no OS-specific permission trickery needed), exercising the
    # best-effort degrade-to-miss/no-op path.
    broken_path = tmp_path / "not_a_file"
    broken_path.mkdir()

    with caplog.at_level(logging.WARNING):
        cache = QueryAnswerCache(str(broken_path), ttl_seconds=3600)
        result = cache.get("any")
        cache.put("any", "text")  # must not raise

    assert result is None
    assert "qa_cache" in caplog.text
