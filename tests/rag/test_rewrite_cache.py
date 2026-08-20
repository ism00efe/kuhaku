"""Tests for the SQLite-backed query-rewrite cache (D48)."""

from __future__ import annotations

from kuhaku.tools.rag import rewrite_cache


def test_put_then_get_within_ttl_returns_the_cached_rewrite(tmp_path):
    db_path = str(tmp_path / "assistant.db")
    rewrite_cache.save_rewrite(db_path, "hash1", "Merkez Bankası politikası", 3600)
    assert rewrite_cache.get_rewrite(db_path, "hash1") == "Merkez Bankası politikası"


def test_get_is_none_for_a_missing_hash(tmp_path):
    db_path = str(tmp_path / "assistant.db")
    assert rewrite_cache.get_rewrite(db_path, "missing") is None


def test_save_rewrite_overwrites_an_existing_hash(tmp_path):
    db_path = str(tmp_path / "assistant.db")
    rewrite_cache.save_rewrite(db_path, "hash1", "first", 3600)
    rewrite_cache.save_rewrite(db_path, "hash1", "second", 3600)
    assert rewrite_cache.get_rewrite(db_path, "hash1") == "second"


def test_get_after_ttl_expiry_returns_none(tmp_path, monkeypatch):
    import kuhaku.tools.rag.rewrite_cache as cache_module

    now = [1_000_000.0]
    monkeypatch.setattr(cache_module.time, "time", lambda: now[0])

    db_path = str(tmp_path / "assistant.db")
    rewrite_cache.save_rewrite(db_path, "hash1", "rewritten", 60)
    assert rewrite_cache.get_rewrite(db_path, "hash1") == "rewritten"

    now[0] += 61  # past the 60s TTL
    assert rewrite_cache.get_rewrite(db_path, "hash1") is None


def test_get_within_ttl_boundary_still_hits(tmp_path, monkeypatch):
    import kuhaku.tools.rag.rewrite_cache as cache_module

    now = [1_000_000.0]
    monkeypatch.setattr(cache_module.time, "time", lambda: now[0])

    db_path = str(tmp_path / "assistant.db")
    rewrite_cache.save_rewrite(db_path, "hash1", "rewritten", 60)

    now[0] += 59  # still inside the window
    assert rewrite_cache.get_rewrite(db_path, "hash1") == "rewritten"


def test_schema_creation_is_idempotent(tmp_path):
    db_path = str(tmp_path / "assistant.db")
    rewrite_cache.save_rewrite(db_path, "hash1", "a", 3600)
    # A second call re-runs _ensure_schema against the same file -- must not raise.
    rewrite_cache.save_rewrite(db_path, "hash2", "b", 3600)
    assert rewrite_cache.get_rewrite(db_path, "hash1") == "a"
    assert rewrite_cache.get_rewrite(db_path, "hash2") == "b"
