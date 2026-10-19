"""Session / conversation memory, independent of any tool.

A :class:`Session` holds one conversation -- its messages, per-tool state and
metadata -- for one owner identity. A :class:`SessionStore` persists it
(:class:`InMemorySessionStore`, :class:`SQLiteSessionStore`) with ownership checks,
expiry and an optimistic version check; :class:`ContextWindow` and the helpers in
:mod:`.window` pick the messages for the next LLM call.

    from kuhaku.core.auth import AuthContext
    from kuhaku.core.models import Message
    from kuhaku.core.session import ContextWindow, InMemorySessionStore

    store = InMemorySessionStore(default_ttl_seconds=3600)
    alice = AuthContext("alice", is_authenticated=True)
    session = store.create(alice)
    session.append(Message.user("hello"))
    store.save(session, alice)
    window = ContextWindow(max_turns=10).apply(store.get(session.id, alice).messages)
"""

from .errors import (
    SessionConflict,
    SessionError,
    SessionFormatError,
    SessionNotFound,
    SessionStoreError,
)
from .models import Session
from .serialization import FORMAT_VERSION, register_part_codec
from .store import (
    BaseSessionStore,
    InMemorySessionStore,
    MessageSanitizer,
    SessionRecord,
    SessionStore,
    SQLiteSessionStore,
    loggable_session_id,
    sanitize_message,
)
from .window import (
    ContextWindow,
    TokenEstimator,
    approximate_tokens,
    last_turns,
    split_turns,
    within_token_budget,
)

__all__ = [
    "FORMAT_VERSION",
    "BaseSessionStore",
    "ContextWindow",
    "InMemorySessionStore",
    "MessageSanitizer",
    "SQLiteSessionStore",
    "Session",
    "SessionConflict",
    "SessionError",
    "SessionFormatError",
    "SessionNotFound",
    "SessionRecord",
    "SessionStore",
    "SessionStoreError",
    "TokenEstimator",
    "approximate_tokens",
    "last_turns",
    "loggable_session_id",
    "register_part_codec",
    "sanitize_message",
    "split_turns",
    "within_token_budget",
]
