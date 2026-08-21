"""SQLite-backed query-answer cache (FR2, Category 2).

Caches only the raw generated ``text: str`` for a given (query, retrieved chunk set,
prompt/LLM/embedding version triple -- D42) -- not the full ``Answer`` object.
``citations``/``retrieved`` are always freshly available in the current request
regardless of hit/miss, and re-running the existing, cheap, deterministic
``RAGEngine._extract_citations`` against the *current* ``retrieved`` avoids ever
replaying stale chunk/citation metadata and needs no new (de)serialization code for the
``Chunk``/``Citation``/``RetrievedChunk`` dataclass trees. See DECISIONS.md D38, D42.

``get``/``put`` are best-effort, mirroring ``llm/token_tracking.py``'s "never break the
caller" idiom: any ``sqlite3.Error`` is caught, logged at WARNING, and treated as a
miss/no-op -- a cache failure must never turn a working answer into a failed request.

Connections are opened per call rather than held open: ``sqlite3.Connection`` is not
thread-safe, and FastAPI dispatches sync route handlers (and therefore this engine) to a
threadpool, so a shared connection would race. WAL mode lets concurrent readers proceed
without blocking on a writer.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Field separator for the cache key payload. Not a character any query, chunk id
# (`doc_id::index`), or prompt version ("v1") can plausibly contain, so it cannot cause
# two distinct inputs to collide onto the same hash.
_KEY_SEP = "\x1f"

# Fallback location when RAGSettings.cache_db_path is unset/empty, so the cache is on by
# default (RAGSettings.cache_enabled=True) without requiring an explicit path -- mirrors
# core/security/audit.py's _DEFAULT_AUDIT_LOG_PATH convention (relative to the process's
# current working directory, created if missing).
_DEFAULT_CACHE_DB_PATH = "./data/kuhaku_qa_cache.sqlite3"


def compute_cache_key(
    query: str,
    chunk_ids: list[str],
    prompt_version: str,
    llm_version: str,
    embedding_version: str,
) -> str:
    """SHA256 hash of ``query`` + the ordered chunk ids + the three deployed versions.

    ``chunk_ids`` order must be preserved exactly as retrieved -- never sorted. ``[S#]``
    citation tags are assigned purely positionally (``rag/prompts.py._format_sources`` /
    ``RAGEngine._extract_citations`` both do ``enumerate(retrieved, start=1)``, with no
    stable per-chunk tag), so a same-set-different-order retrieval must produce a
    different cache key -- otherwise a cached answer's ``[S#]`` tags could be replayed
    against a reordered context and point at the wrong sources.

    ``prompt_version``/``llm_version``/``embedding_version`` (D42) are the deployed
    versions from ``Settings``/``RAGSettings`` -- changing any of them changes the hash,
    so upgrading a model or the system prompt automatically invalidates every previously
    cached answer rather than silently serving a response a different configuration
    produced.
    """

    payload = _KEY_SEP.join(
        [query, *chunk_ids, prompt_version, llm_version, embedding_version]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QueryAnswerCache:
    """Get/put the cached answer text for a cache key, with a TTL.

    ``db_path`` empty/``None`` (``RAGSettings.cache_db_path``'s own default) falls back
    to :data:`_DEFAULT_CACHE_DB_PATH`.

    Deliberately touches no filesystem state in ``__init__`` -- schema creation is
    lazy, on first :meth:`get`/:meth:`put` (see ``_ensure_schema``). ``RAG.__init__``
    builds one of these unconditionally whenever ``RAGSettings.cache_enabled`` is true
    (its default), so *constructing* a cache must not itself be a guaranteed disk
    write: plenty of ``RAG()`` callers (including ones that never call ``ask()``, or
    whose every request abstains before the cache lookup even runs) would otherwise
    create a database file they never use. This mirrors ``audit_log_path``'s own "no
    write until there's something to write" behavior.
    """

    def __init__(self, db_path: str, ttl_seconds: int) -> None:
        self._db_path = db_path or _DEFAULT_CACHE_DB_PATH
        self._ttl_seconds = ttl_seconds
        self._schema_ready = False

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> bool:
        """Create the table on first use, memoized so steady-state get/put calls skip
        straight to their own query. Returns whether the schema is usable -- ``False``
        leaves ``_schema_ready`` unset so the next call retries (a transient failure,
        e.g. a directory that becomes writable later, self-heals)."""

        if self._schema_ready:
            return True

        # sqlite3.connect() does not create missing parent directories (unlike
        # security/audit.py's file-append path) -- do that first so a fresh default
        # location (./data/) works out of the box. A read-only/unwritable parent must
        # degrade to no caching, not fail the caller -- same idiom as the sqlite3.Error
        # catch below.
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning(
                "could not create qa_cache directory; caching disabled for this process",
                exc_info=True,
            )
            return False

        try:
            with self._connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS qa_cache ("
                    "cache_key TEXT PRIMARY KEY, "
                    "answer_text TEXT NOT NULL, "
                    "created_at REAL NOT NULL"
                    ")"
                )
        except sqlite3.Error:
            logger.warning(
                "qa_cache schema init failed; caching disabled for this process",
                exc_info=True,
            )
            return False

        self._schema_ready = True
        return True

    def get(self, cache_key: str) -> str | None:
        """Return the cached answer text, or ``None`` on a miss/expiry/failure."""

        if not self._ensure_schema():
            return None

        try:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT answer_text, created_at FROM qa_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        except sqlite3.Error:
            logger.warning("qa_cache read failed; treating as a miss", exc_info=True)
            return None

        if row is None:
            return None

        answer_text, created_at = row
        if time.time() - created_at > self._ttl_seconds:
            return None
        return str(answer_text)

    def put(self, cache_key: str, answer_text: str) -> None:
        """Best-effort write; a failure here must never affect the caller's response."""

        if not self._ensure_schema():
            return

        try:
            with self._connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO qa_cache (cache_key, answer_text, created_at) "
                    "VALUES (?, ?, ?)",
                    (cache_key, answer_text, time.time()),
                )
        except sqlite3.Error:
            logger.warning("qa_cache write failed; answer was not cached", exc_info=True)
