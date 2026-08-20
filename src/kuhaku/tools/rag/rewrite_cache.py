"""SQLite-backed cache for LLM query rewrites (D48).

Mirrors ``app_config.py``/``evaluation/storage.py``'s connection idiom: a short-lived
per-call connection (never held open), ``PRAGMA journal_mode=WAL``, idempotent
``CREATE TABLE IF NOT EXISTS``. Lives in the same ``settings.sqlite_db_path`` file as
``qa_cache``/``feedback``/``app_config``/the D47 evaluation tables.

``get_rewrite``/``save_rewrite`` are best-effort (mirrors ``rag/cache.py``'s QA-cache
idiom, not ``evaluation/storage.py``'s "let it raise" one): a rewrite-cache failure must
degrade to "call the LLM again" or "skip caching", never turn a working answer into a
failed request.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def _connection(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rewrite_cache ("
        "query_hash TEXT PRIMARY KEY, "
        "rewritten_query TEXT NOT NULL, "
        "created_at REAL NOT NULL, "
        "ttl_seconds INTEGER NOT NULL"
        ")"
    )


def get_rewrite(db_path: str, query_hash: str) -> str | None:
    """Return the cached rewrite for ``query_hash``, or ``None`` on a miss/expiry/failure."""

    try:
        with _connection(db_path) as conn:
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT rewritten_query, created_at, ttl_seconds "
                "FROM rewrite_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
    except sqlite3.Error:
        logger.warning("rewrite_cache read failed; treating as a miss", exc_info=True)
        return None

    if row is None:
        return None

    rewritten_query, created_at, ttl_seconds = row
    if time.time() - created_at > ttl_seconds:
        return None
    return str(rewritten_query)


def save_rewrite(db_path: str, query_hash: str, rewritten_query: str, ttl_seconds: int) -> None:
    """Best-effort write; a failure here must never affect the caller's response."""

    try:
        with _connection(db_path) as conn:
            _ensure_schema(conn)
            conn.execute(
                "INSERT OR REPLACE INTO rewrite_cache "
                "(query_hash, rewritten_query, created_at, ttl_seconds) VALUES (?, ?, ?, ?)",
                (query_hash, rewritten_query, time.time(), ttl_seconds),
            )
    except sqlite3.Error:
        logger.warning("rewrite_cache write failed; rewrite was not cached", exc_info=True)
