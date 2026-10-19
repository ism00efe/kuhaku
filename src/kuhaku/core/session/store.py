"""Session persistence: the :class:`SessionStore` protocol and two backends.

Both backends share one implementation of the rules (:class:`BaseSessionStore`) and
differ only in five record-level primitives, so ownership, expiry, the optimistic
version check and the sanitizer hook behave identically whichever is used:

- :class:`InMemorySessionStore` -- a dict behind a lock; for tests and single-process
  use. Records are kept serialized, so a loaded session never aliases stored state.
- :class:`SQLiteSessionStore` -- stdlib ``sqlite3``, one file.

What is stored: every message as given (including raw user text, which may contain
personal data), each tool's state (an extraction tool stores the values it extracted),
the session metadata, the owner's identity string and the timestamps. Nothing is
stored until :meth:`SessionStore.save`; nothing is removed except by
:meth:`~SessionStore.delete`, by :meth:`~SessionStore.purge_expired`, or -- for reads --
by expiry, which hides a session immediately even before it is purged.

Messages are sanitized before they are written: by default with
:func:`sanitize_message` (the same PII rules as the rest of kuhaku), and the string
leaves of the session ``metadata`` with :func:`kuhaku.core.sanitization.sanitize`. Pass
your own ``sanitizer=`` to change how messages are cleaned, or ``sanitizer=None`` to
store messages and metadata as given -- for a tool that needs a value from an earlier
turn verbatim (a phone number the user gave, say).

Tool ``state`` is never sanitized: it holds the tool's own results (the extraction tool
stores the values it extracted, and the next turn continues from them), so masking it
would change what the tool reads back. A tool that keeps personal data in its state is
responsible for it.
"""

from __future__ import annotations

import logging
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..auth import AuthContext
from ..exceptions import ConfigError
from ..models import ContentPart, Message, TextPart, ToolResultPart
from ..observability import instrumented_step, record_redactions
from ..sanitization import sanitize, sanitize_text
from .errors import SessionConflict, SessionNotFound, SessionStoreError
from .models import Session
from .serialization import decode_payload, encode_payload

logger = logging.getLogger(__name__)

MessageSanitizer = Callable[[Message], Message]
"""Applied to every message on save; its return value is what is stored."""

Clock = Callable[[], float]


@runtime_checkable
class SessionStore(Protocol):
    """Creates, loads, saves and deletes sessions on behalf of an identity.

    Every method that addresses a session takes the caller's :class:`AuthContext`; a
    session belonging to another identity is reported exactly like a missing one
    (:class:`SessionNotFound`).
    """

    def create(
        self,
        auth: AuthContext,
        *,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session: ...

    def get(self, session_id: str, auth: AuthContext) -> Session: ...

    def save(self, session: Session, auth: AuthContext) -> Session: ...

    def delete(self, session_id: str, auth: AuthContext) -> bool: ...

    def purge_expired(self) -> int: ...


@dataclass(frozen=True)
class SessionRecord:
    """A session as a backend stores it: the columns access checks need, plus the
    JSON body (messages, tool state, metadata)."""

    id: str
    owner: str
    authenticated: bool
    created_at: float
    updated_at: float
    version: int
    expires_at: float | None
    ttl_seconds: float | None
    payload: str


_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{24}")


def new_session_id() -> str:
    # Always minted here, never caller-supplied: for an anonymous owner the id is the
    # only thing standing between one anonymous caller and another's session, so it
    # must be unguessable. URL-safe base64 rather than hex so it is not mistaken for
    # (and masked as) a secret by the sanitizer's long-hex rule when it appears in text.
    return secrets.token_urlsafe(18)


def loggable_session_id(session_id: object) -> str:
    """``session_id`` if it has the shape of a minted id, else ``"invalid"``.

    Ids passed to :meth:`SessionStore.get`/``delete`` come from the caller and could be
    any text; only one that looks like an id kuhaku minted is put in a log record or
    span, so the session_id field can never carry user content.
    """

    if isinstance(session_id, str) and _SESSION_ID_RE.fullmatch(session_id):
        return session_id
    return "invalid"


def sanitize_message(message: Message) -> Message:
    """A :data:`MessageSanitizer` that masks PII in text with
    :func:`kuhaku.core.sanitization.sanitize_text`.

    Text parts (and plain-string content) and tool results' content are masked; other
    part types pass through unchanged -- they are not text this sanitizer understands
    (tool-call arguments are structured data, images and documents are bytes).
    Message metadata is not touched. This is a store's default ``sanitizer``.
    """

    redactions = []

    def clean(text: str) -> str:
        cleaned, found = sanitize_text(text)
        redactions.extend(found)
        return cleaned

    if isinstance(message.content, str):
        result = Message(message.role, clean(message.content), dict(message.metadata))
    else:
        parts: list[ContentPart] = []
        for part in message.parts:
            if isinstance(part, TextPart):
                parts.append(TextPart(clean(part.text)))
            elif isinstance(part, ToolResultPart):
                parts.append(replace(part, content=clean(part.content)))
            else:
                parts.append(part)
        result = Message(message.role, parts, dict(message.metadata))
    if redactions:
        record_redactions(redactions)
    return result


class BaseSessionStore:
    """The session rules over five record primitives a backend implements.

    ``default_ttl_seconds`` applies to sessions created without their own
    ``ttl_seconds``; ``None`` means sessions do not expire. Expiry slides: every save
    moves ``expires_at`` to ``now + ttl``, with the TTL the session was created with
    (read from the store, not from the working copy). ``clock`` returns seconds since
    the epoch (injectable for tests). ``sanitizer`` -- see the module docstring; on by
    default, ``None`` turns it off.
    """

    def __init__(
        self,
        *,
        default_ttl_seconds: float | None = None,
        sanitizer: MessageSanitizer | None = sanitize_message,
        clock: Clock = time.time,
    ) -> None:
        _check_ttl(default_ttl_seconds)
        self._default_ttl = default_ttl_seconds
        self._sanitizer = sanitizer
        self._clock = clock

    # --- backend primitives -----------------------------------------------------------

    def _insert(self, record: SessionRecord) -> None:
        raise NotImplementedError

    def _load(self, session_id: str) -> SessionRecord | None:
        raise NotImplementedError

    def _replace(self, record: SessionRecord, *, expected_version: int, now: float) -> bool:
        """Atomically store ``record`` iff the stored row has the same id and owner, is
        unexpired at ``now``, and is at ``expected_version``. True when stored."""

        raise NotImplementedError

    def _remove(self, session_id: str, owner: str, authenticated: bool) -> bool:
        raise NotImplementedError

    def _remove_expired(self, now: float) -> int:
        raise NotImplementedError

    # --- the SessionStore API ---------------------------------------------------------

    def create(
        self,
        auth: AuthContext,
        *,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        _check_auth(auth)
        _check_ttl(ttl_seconds)
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        now = self._clock()
        session = Session(
            id=new_session_id(),
            owner=auth.identity,
            authenticated=auth.is_authenticated,
            created_at=now,
            updated_at=now,
            version=1,
            expires_at=None if ttl is None else now + ttl,
            ttl_seconds=ttl,
            metadata=self._clean_metadata(dict(metadata or {})),
        )
        with instrumented_step("session.create", logger, session_id=session.id):
            self._insert(_to_record(session, encode_payload([], {}, session.metadata)))
        return session

    def get(self, session_id: str, auth: AuthContext) -> Session:
        _check_auth(auth)
        with instrumented_step(
            "session.get", logger, session_id=loggable_session_id(session_id)
        ) as rec:
            record = self._load(session_id)
            if record is None or not _visible(record, auth, self._clock()):
                raise SessionNotFound(session_id)
            session = _from_record(record)
            rec.set(version=session.version)
            return session

    def save(self, session: Session, auth: AuthContext) -> Session:
        """Persist ``session`` and return it, updated in place: ``version`` bumped,
        ``updated_at``/``expires_at`` moved, and -- with a sanitizer -- ``messages``
        replaced by what was actually stored.

        Raises :class:`SessionConflict` when another save landed since this copy was
        loaded, and :class:`SessionNotFound` when the session is gone, expired, or not
        ``auth``'s.
        """

        _check_auth(auth)
        with instrumented_step(
            "session.save", logger, session_id=loggable_session_id(session.id)
        ) as rec:
            now = self._clock()
            if not session.is_owned_by(auth):
                raise SessionNotFound(session.id)
            # The TTL is the stored one: a working copy whose ttl_seconds was edited
            # (set to None, say) must not make the session immortal.
            current = self._load(session.id)
            if current is None or not _visible(current, auth, now):
                raise SessionNotFound(session.id)
            ttl = current.ttl_seconds
            messages = list(session.messages)
            if self._sanitizer is not None:
                messages = [self._sanitizer(message) for message in messages]
            metadata = self._clean_metadata(session.metadata)
            payload = encode_payload(messages, session.state, metadata)
            updated = replace(
                session,
                version=session.version + 1,
                updated_at=now,
                expires_at=None if ttl is None else now + ttl,
                ttl_seconds=ttl,
            )
            stored = self._replace(
                _to_record(updated, payload), expected_version=session.version, now=now
            )
            if not stored:
                current = self._load(session.id)
                if current is None or not _visible(current, auth, now):
                    raise SessionNotFound(session.id)
                raise SessionConflict(
                    session.id, expected=session.version, actual=current.version
                )
            session.version = updated.version
            session.updated_at = updated.updated_at
            session.expires_at = updated.expires_at
            session.ttl_seconds = ttl
            session.messages = messages
            session.metadata = metadata
            rec.set(version=session.version)
            return session

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        if self._sanitizer is None:
            return metadata
        return _sanitize_leaves(metadata)

    def delete(self, session_id: str, auth: AuthContext) -> bool:
        """Delete the session if ``auth`` owns it. False -- not an error -- when there
        is nothing of ``auth``'s to delete, so deleting twice is harmless."""

        _check_auth(auth)
        with instrumented_step(
            "session.delete", logger, session_id=loggable_session_id(session_id)
        ):
            return self._remove(session_id, auth.identity, auth.is_authenticated)

    def purge_expired(self) -> int:
        """Physically remove every expired session; returns how many. Expired sessions
        are already invisible to :meth:`get` -- this reclaims their storage."""

        with instrumented_step("session.purge", logger) as rec:
            removed = self._remove_expired(self._clock())
            rec.set(result_count=removed)
            return removed


class InMemorySessionStore(BaseSessionStore):
    """Process-local, thread-safe store. Everything is lost when the process exits."""

    def __init__(
        self,
        *,
        default_ttl_seconds: float | None = None,
        sanitizer: MessageSanitizer | None = sanitize_message,
        clock: Clock = time.time,
    ) -> None:
        super().__init__(
            default_ttl_seconds=default_ttl_seconds, sanitizer=sanitizer, clock=clock
        )
        self._records: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def _insert(self, record: SessionRecord) -> None:
        with self._lock:
            self._records[record.id] = record

    def _load(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._records.get(session_id)

    def _replace(self, record: SessionRecord, *, expected_version: int, now: float) -> bool:
        with self._lock:
            current = self._records.get(record.id)
            if (
                current is None
                or current.version != expected_version
                or current.owner != record.owner
                or current.authenticated != record.authenticated
                or _expired(current, now)
            ):
                return False
            self._records[record.id] = record
            return True

    def _remove(self, session_id: str, owner: str, authenticated: bool) -> bool:
        with self._lock:
            current = self._records.get(session_id)
            if current is None or current.owner != owner or current.authenticated != authenticated:
                return False
            del self._records[session_id]
            return True

    def _remove_expired(self, now: float) -> int:
        with self._lock:
            expired = [sid for sid, record in self._records.items() if _expired(record, now)]
            for sid in expired:
                del self._records[sid]
            return len(expired)


class SQLiteSessionStore(BaseSessionStore):
    """Sessions in one SQLite file (stdlib ``sqlite3``).

    Connections are opened per call, as in the RAG query-answer cache: a
    ``sqlite3.Connection`` is not thread-safe, and WAL mode lets readers proceed while a
    writer commits. The version check is a single ``UPDATE ... WHERE version = ?``, so
    it is atomic across threads and processes sharing the file.

    The schema is created lazily on first use, so constructing a store writes nothing.
    Unlike that cache, a failure is raised (:class:`SessionStoreError`), never swallowed.
    ``":memory:"`` is refused: with a connection per call every call would see a new,
    empty database -- use :class:`InMemorySessionStore`.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        default_ttl_seconds: float | None = None,
        sanitizer: MessageSanitizer | None = sanitize_message,
        clock: Clock = time.time,
    ) -> None:
        super().__init__(
            default_ttl_seconds=default_ttl_seconds, sanitizer=sanitizer, clock=clock
        )
        if not str(db_path) or str(db_path) == ":memory:":
            raise ConfigError(
                "SQLiteSessionStore needs a file path; for a process-local store use "
                "InMemorySessionStore."
            )
        self._db_path = str(db_path)
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._ensure_schema()
        try:
            with self._connect() as conn:
                yield conn
        except sqlite3.Error as exc:
            raise SessionStoreError(f"Session store operation failed: {exc}.") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            try:
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
                with self._connect() as conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS kuhaku_sessions ("
                        "id TEXT PRIMARY KEY, "
                        "owner TEXT NOT NULL, "
                        "authenticated INTEGER NOT NULL, "
                        "created_at REAL NOT NULL, "
                        "updated_at REAL NOT NULL, "
                        "version INTEGER NOT NULL, "
                        "expires_at REAL, "
                        "ttl_seconds REAL, "
                        "payload TEXT NOT NULL"
                        ")"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS kuhaku_sessions_expires_at "
                        "ON kuhaku_sessions (expires_at)"
                    )
            except (OSError, sqlite3.Error) as exc:
                raise SessionStoreError(
                    f"Could not initialize the session store at {self._db_path}: {exc}."
                ) from exc
            self._schema_ready = True

    def _insert(self, record: SessionRecord) -> None:
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO kuhaku_sessions (id, owner, authenticated, created_at, "
                "updated_at, version, expires_at, ttl_seconds, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.owner,
                    int(record.authenticated),
                    record.created_at,
                    record.updated_at,
                    record.version,
                    record.expires_at,
                    record.ttl_seconds,
                    record.payload,
                ),
            )

    def _load(self, session_id: str) -> SessionRecord | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id, owner, authenticated, created_at, updated_at, version, "
                "expires_at, ttl_seconds, payload FROM kuhaku_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionRecord(
            id=row[0],
            owner=row[1],
            authenticated=bool(row[2]),
            created_at=row[3],
            updated_at=row[4],
            version=row[5],
            expires_at=row[6],
            ttl_seconds=row[7],
            payload=row[8],
        )

    def _replace(self, record: SessionRecord, *, expected_version: int, now: float) -> bool:
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE kuhaku_sessions SET updated_at = ?, version = ?, expires_at = ?, "
                "payload = ? WHERE id = ? AND owner = ? AND authenticated = ? "
                "AND version = ? AND (expires_at IS NULL OR expires_at > ?)",
                (
                    record.updated_at,
                    record.version,
                    record.expires_at,
                    record.payload,
                    record.id,
                    record.owner,
                    int(record.authenticated),
                    expected_version,
                    now,
                ),
            )
            return cursor.rowcount == 1

    def _remove(self, session_id: str, owner: str, authenticated: bool) -> bool:
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM kuhaku_sessions WHERE id = ? AND owner = ? AND authenticated = ?",
                (session_id, owner, int(authenticated)),
            )
            return cursor.rowcount == 1

    def _remove_expired(self, now: float) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM kuhaku_sessions WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            return cursor.rowcount


def _sanitize_leaves(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, dict):
        return {key: _sanitize_leaves(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_leaves(item) for item in value]
    return value


def _check_auth(auth: AuthContext) -> None:
    if not isinstance(auth, AuthContext):
        raise TypeError(f"auth must be an AuthContext, got {type(auth).__name__}.")


def _check_ttl(ttl: float | None) -> None:
    if ttl is not None and not ttl > 0:
        raise ConfigError(f"A session TTL must be a positive number of seconds; got {ttl!r}.")


def _expired(record: SessionRecord, now: float) -> bool:
    return record.expires_at is not None and record.expires_at <= now


def _visible(record: SessionRecord, auth: AuthContext, now: float) -> bool:
    return (
        record.owner == auth.identity
        and record.authenticated == auth.is_authenticated
        and not _expired(record, now)
    )


def _to_record(session: Session, payload: str) -> SessionRecord:
    return SessionRecord(
        id=session.id,
        owner=session.owner,
        authenticated=session.authenticated,
        created_at=session.created_at,
        updated_at=session.updated_at,
        version=session.version,
        expires_at=session.expires_at,
        ttl_seconds=session.ttl_seconds,
        payload=payload,
    )


def _from_record(record: SessionRecord) -> Session:
    messages, state, metadata = decode_payload(record.payload)
    return Session(
        id=record.id,
        owner=record.owner,
        authenticated=record.authenticated,
        created_at=record.created_at,
        updated_at=record.updated_at,
        version=record.version,
        expires_at=record.expires_at,
        ttl_seconds=record.ttl_seconds,
        messages=messages,
        state=state,
        metadata=metadata,
    )
