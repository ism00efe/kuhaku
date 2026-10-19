"""Errors the session layer raises."""

from __future__ import annotations

import logging

from ..exceptions import KuhakuError


class SessionError(KuhakuError):
    """Root of every session-layer error."""


class SessionNotFound(SessionError):
    """No session with this id is visible to the caller.

    Raised alike for a session that never existed, one that expired, one that was
    deleted, and one that belongs to a different identity -- the message is the same in
    every case, so a caller cannot probe for other people's session ids. Mirrors the RAG
    rule that an access denial is indistinguishable from absence.

    The message does not quote the id -- it is caller-supplied text and the message ends
    up in logs; ``session_id`` holds it for code that needs it. An expected outcome, so
    the step log records it at INFO, not ERROR.
    """

    log_level = logging.INFO

    def __init__(self, session_id: str) -> None:
        super().__init__("No such session was found.")
        self.session_id = session_id


class SessionConflict(SessionError):
    """The session changed in the store since this copy was loaded.

    Two turns that load the same session concurrently both hold version ``n``; the first
    save makes it ``n + 1`` and the second gets this error instead of silently
    overwriting the first turn. Reload with ``get`` and redo the turn.
    """

    def __init__(self, session_id: str, *, expected: int, actual: int) -> None:
        super().__init__(
            f"Session '{session_id}' was modified concurrently: this copy is version "
            f"{expected}, the store has version {actual}. Reload it and retry the turn."
        )
        self.session_id = session_id
        self.expected = expected
        self.actual = actual


class SessionFormatError(SessionError):
    """A stored session cannot be decoded (or a session cannot be encoded): an unknown
    format version, a content part type with no registered codec, or data that is not
    JSON-serializable."""


class SessionStoreError(SessionError):
    """The storage backend failed (e.g. a SQLite error). Unlike the query-answer cache,
    a session store never degrades silently: losing a turn is not a cache miss."""
