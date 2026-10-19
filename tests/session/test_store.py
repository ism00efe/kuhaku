"""Session stores: ownership, expiry, optimistic concurrency, sanitizer hook -- run
against both backends, so the two cannot drift apart."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from kuhaku.core.auth import AuthContext
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.models import Message, TextPart
from kuhaku.core.session import (
    InMemorySessionStore,
    SessionConflict,
    SessionFormatError,
    SessionNotFound,
    SessionStore,
    SQLiteSessionStore,
    loggable_session_id,
    sanitize_message,
)

ALICE = AuthContext("alice", is_authenticated=True)
BOB = AuthContext("bob", is_authenticated=True)


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
        return SQLiteSessionStore(tmp_path / "nested" / "sessions.sqlite3", **kwargs)

    return build


def test_both_backends_satisfy_the_protocol(make_store):
    assert isinstance(make_store(), SessionStore)


def test_create_save_get_round_trips_messages_state_and_metadata(make_store):
    store = make_store()
    session = store.create(ALICE, metadata={"channel": "web"})
    session.append(Message.user("hello"), Message("assistant", [TextPart("hi "), TextPart("there")]))
    session.tool_state("extraction")["Order"] = {"data": {"product": "tea"}}
    store.save(session, ALICE)

    loaded = store.get(session.id, ALICE)
    assert loaded.messages == session.messages
    assert loaded.state == {"extraction": {"Order": {"data": {"product": "tea"}}}}
    assert loaded.metadata == {"channel": "web"}
    assert loaded.owner == "alice" and loaded.authenticated
    assert loaded.version == session.version == 2


def test_loaded_sessions_never_alias_stored_state(make_store):
    store = make_store()
    session = store.create(ALICE)
    store.save(session, ALICE)
    loaded = store.get(session.id, ALICE)
    loaded.append(Message.user("unsaved"))
    loaded.tool_state("x")["k"] = 1

    again = store.get(session.id, ALICE)
    assert again.messages == [] and again.state == {}


def test_nothing_is_stored_until_create(make_store, tmp_path):
    make_store()
    assert not (tmp_path / "nested").exists()


def test_another_identity_gets_not_found_exactly_like_a_missing_id(make_store):
    store = make_store()
    session = store.create(ALICE)

    with pytest.raises(SessionNotFound) as denied:
        store.get(session.id, BOB)
    with pytest.raises(SessionNotFound) as missing:
        store.get("x" * 24, BOB)
    assert type(denied.value) is type(missing.value)
    assert str(denied.value).replace(session.id, "ID") == str(missing.value).replace("x" * 24, "ID")


def test_unauthenticated_context_with_the_same_identity_is_a_different_owner(make_store):
    store = make_store()
    session = store.create(ALICE)
    with pytest.raises(SessionNotFound):
        store.get(session.id, AuthContext("alice", is_authenticated=False))


def test_save_and_delete_by_another_identity_are_refused_as_absence(make_store):
    store = make_store()
    session = store.create(ALICE)

    with pytest.raises(SessionNotFound):
        store.save(session, BOB)
    assert store.delete(session.id, BOB) is False
    assert store.get(session.id, ALICE).id == session.id


def test_a_forged_session_object_cannot_overwrite_someone_elses(make_store):
    store = make_store()
    session = store.create(ALICE)
    forged = store.get(session.id, ALICE)
    forged.owner, forged.authenticated = "bob", True
    with pytest.raises(SessionNotFound):
        store.save(forged, BOB)


def test_delete_is_idempotent(make_store):
    store = make_store()
    session = store.create(ALICE)
    assert store.delete(session.id, ALICE) is True
    assert store.delete(session.id, ALICE) is False
    with pytest.raises(SessionNotFound):
        store.get(session.id, ALICE)


def test_concurrent_turns_conflict_instead_of_overwriting(make_store):
    store = make_store()
    session = store.create(ALICE)
    first = store.get(session.id, ALICE)
    second = store.get(session.id, ALICE)

    first.append(Message.user("turn one"))
    store.save(first, ALICE)
    second.append(Message.user("turn two"))
    with pytest.raises(SessionConflict) as conflict:
        store.save(second, ALICE)

    assert (conflict.value.expected, conflict.value.actual) == (1, 2)
    assert [m.text for m in store.get(session.id, ALICE).messages] == ["turn one"]


def test_racing_threads_lose_no_turn_and_each_conflict_is_reported(make_store):
    store = make_store()
    session = store.create(ALICE)
    outcomes: list[str] = []
    barrier = threading.Barrier(8)

    def turn(i: int) -> None:
        copy = store.get(session.id, ALICE)
        copy.append(Message.user(f"turn {i}"))
        barrier.wait()
        try:
            store.save(copy, ALICE)
            outcomes.append("saved")
        except SessionConflict:
            outcomes.append("conflict")

    threads = [threading.Thread(target=turn, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("saved") == 1
    assert outcomes.count("conflict") == 7
    assert len(store.get(session.id, ALICE).messages) == 1


def test_ttl_expiry_hides_then_purge_removes(make_store):
    clock = FakeClock()
    store = make_store(default_ttl_seconds=60, clock=clock)
    session = store.create(ALICE)
    assert session.expires_at == 1_060

    clock.now = 1_059
    store.get(session.id, ALICE)
    clock.now = 1_060
    with pytest.raises(SessionNotFound):
        store.get(session.id, ALICE)
    with pytest.raises(SessionNotFound):
        store.save(session, ALICE)
    assert store.purge_expired() == 1
    assert store.purge_expired() == 0


def test_expiry_slides_on_save_and_per_session_ttl_overrides_default(make_store):
    clock = FakeClock()
    store = make_store(default_ttl_seconds=60, clock=clock)
    session = store.create(ALICE, ttl_seconds=10)
    clock.now = 1_005
    store.save(session, ALICE)
    assert session.expires_at == 1_015
    clock.now = 1_012
    assert store.get(session.id, ALICE).version == 2


def test_no_ttl_means_no_expiry(make_store):
    clock = FakeClock()
    store = make_store(clock=clock)
    session = store.create(ALICE)
    clock.now = 10**12
    assert store.get(session.id, ALICE).expires_at is None
    assert store.purge_expired() == 0


@pytest.mark.parametrize("ttl", [0, -1])
def test_non_positive_ttl_is_a_config_error(make_store, ttl):
    with pytest.raises(ConfigError):
        make_store(default_ttl_seconds=ttl)
    with pytest.raises(ConfigError):
        make_store().create(ALICE, ttl_seconds=ttl)


def test_sanitizer_hook_controls_what_is_stored(make_store):
    store = make_store(sanitizer=sanitize_message)
    session = store.create(ALICE)
    session.append(Message.user("mail me at someone@example.com"))
    store.save(session, ALICE)

    assert session.messages[0].text == "mail me at [EMAIL]"
    assert store.get(session.id, ALICE).messages[0].text == "mail me at [EMAIL]"


def test_without_a_sanitizer_text_is_stored_as_given(make_store):
    store = make_store(sanitizer=None)  # sanitizing is the default now; opt out explicitly
    session = store.create(ALICE)
    session.append(Message.user("someone@example.com"))
    store.save(session, ALICE)
    assert store.get(session.id, ALICE).messages[0].text == "someone@example.com"


def test_non_json_state_fails_at_save_and_leaves_the_store_unchanged(make_store):
    store = make_store()
    session = store.create(ALICE)
    session.tool_state("x")["bad"] = object()
    with pytest.raises(SessionFormatError):
        store.save(session, ALICE)
    assert store.get(session.id, ALICE).version == 1


def test_auth_must_be_an_auth_context(make_store):
    with pytest.raises(TypeError):
        make_store().create("alice")  # type: ignore[arg-type]


def test_session_ids_are_unique_and_only_minted_shapes_are_loggable(make_store):
    store = make_store()
    ids = {store.create(ALICE).id for _ in range(50)}
    assert len(ids) == 50
    assert all(loggable_session_id(sid) == sid for sid in ids)
    assert loggable_session_id("alice's credit card 4111") == "invalid"
    assert loggable_session_id(None) == "invalid"


def test_sqlite_store_refuses_memory_path():
    with pytest.raises(ConfigError):
        SQLiteSessionStore(":memory:")


def test_sqlite_sessions_survive_a_new_store_instance(tmp_path):
    path = tmp_path / "s.sqlite3"
    session = SQLiteSessionStore(path).create(ALICE)
    session.append(Message.user("persisted"))
    SQLiteSessionStore(path).save(session, ALICE)
    assert SQLiteSessionStore(path).get(session.id, ALICE).messages[0].text == "persisted"


def test_sqlite_row_from_a_newer_format_is_refused_with_a_clear_error(tmp_path):
    path = tmp_path / "s.sqlite3"
    store = SQLiteSessionStore(path)
    session = store.create(ALICE)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE kuhaku_sessions SET payload = ?", ('{"format": 99}',))
    conn.close()
    with pytest.raises(SessionFormatError, match="newer kuhaku"):
        store.get(session.id, ALICE)


def test_session_operations_log_the_id_but_never_message_content(make_store, caplog):
    store = make_store()
    with caplog.at_level("INFO"):
        session = store.create(ALICE)
        session.append(Message.user("secret-text-xyz"))
        store.save(session, ALICE)
        store.get(session.id, ALICE)

    records = [r for r in caplog.records if getattr(r, "step", "").startswith("session.")]
    assert {r.step for r in records} >= {"session.create", "session.save", "session.get"}
    assert all(r.session_id == session.id for r in records)
    assert "secret-text-xyz" not in caplog.text
