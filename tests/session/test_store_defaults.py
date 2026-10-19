"""Session stores sanitize by default (messages and metadata, not tool state), can be
told not to, take the TTL from the stored record, and reject malformed stored state."""

from __future__ import annotations

import pytest

from kuhaku.core.auth import AuthContext
from kuhaku.core.models import Message
from kuhaku.core.session import (
    InMemorySessionStore,
    SessionFormatError,
    SQLiteSessionStore,
)
from kuhaku.core.session.serialization import decode_payload

ALICE = AuthContext("alice", is_authenticated=True)


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture(params=["memory", "sqlite"])
def make_store(request, tmp_path):
    def build(**kwargs):
        if request.param == "memory":
            return InMemorySessionStore(**kwargs)
        return SQLiteSessionStore(tmp_path / "sessions.sqlite3", **kwargs)

    return build


def test_messages_and_metadata_are_sanitized_by_default(make_store):
    store = make_store()
    session = store.create(ALICE, metadata={"contact": "ali@example.com"})
    session.append(Message.user("call me on +90 532 123 45 67"))
    session.metadata["notes"] = ["password: hunter2"]
    session.tool_state("extraction")["Contact"] = {"data": {"phone": "+90 532 123 45 67"}}
    store.save(session, ALICE)

    stored = store.get(session.id, ALICE)
    assert stored.messages[0].text == "call me on [PHONE]"
    assert stored.metadata == {"contact": "[EMAIL]", "notes": ["password: [SECRET]"]}
    # Tool state is the tool's own data and is kept verbatim.
    assert stored.state["extraction"]["Contact"]["data"]["phone"] == "+90 532 123 45 67"
    # The working copy reflects what was stored.
    assert session.messages[0].text == "call me on [PHONE]"


def test_sanitizer_none_stores_as_given(make_store):
    store = make_store(sanitizer=None)
    session = store.create(ALICE, metadata={"contact": "ali@example.com"})
    session.append(Message.user("ali@example.com"))
    store.save(session, ALICE)
    stored = store.get(session.id, ALICE)
    assert stored.messages[0].text == "ali@example.com"
    assert stored.metadata == {"contact": "ali@example.com"}


def test_custom_sanitizer_replaces_the_message_rule(make_store):
    store = make_store(sanitizer=lambda m: Message(m.role, m.text.upper(), dict(m.metadata)))
    session = store.create(ALICE)
    session.append(Message.user("hello"))
    store.save(session, ALICE)
    assert store.get(session.id, ALICE).messages[0].text == "HELLO"


def test_ttl_comes_from_the_stored_record_not_the_working_copy(make_store):
    clock = FakeClock()
    store = make_store(default_ttl_seconds=60, clock=clock)
    session = store.create(ALICE)
    session.ttl_seconds = None  # trying to make it immortal
    store.save(session, ALICE)
    assert session.ttl_seconds == 60
    assert session.expires_at == clock.now + 60
    clock.now += 61
    with pytest.raises(Exception, match="No such session"):
        store.get(session.id, ALICE)


@pytest.mark.parametrize(
    "payload",
    [
        '{"format": 1, "messages": [], "state": [1, 2], "metadata": {}}',
        '{"format": 1, "messages": [], "state": {}, "metadata": "x"}',
        '{"format": 1, "messages": ["not a message"], "state": {}, "metadata": {}}',
    ],
)
def test_malformed_stored_state_is_a_format_error(payload):
    with pytest.raises(SessionFormatError):
        decode_payload(payload)
