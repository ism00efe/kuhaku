"""Step logs and spans carry an exception's type, never its message; expected
outcomes (session not found / conflict) are not logged at ERROR."""

from __future__ import annotations

import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from kuhaku.core.auth import AuthContext
from kuhaku.core.observability import tracing
from kuhaku.core.session import InMemorySessionStore, SessionNotFound

SECRET_ID = "alice@example.com-not-an-id"


def test_session_not_found_is_info_and_does_not_echo_the_id(caplog):
    store = InMemorySessionStore()
    alice = AuthContext("alice", is_authenticated=True)
    with caplog.at_level(logging.DEBUG), pytest.raises(SessionNotFound) as info:
        store.get(SECRET_ID, alice)
    assert SECRET_ID not in str(info.value)
    assert info.value.session_id == SECRET_ID
    step_records = [r for r in caplog.records if getattr(r, "step", None) == "session.get"]
    assert step_records and all(r.levelno < logging.ERROR for r in step_records)
    assert all(SECRET_ID not in r.getMessage() for r in caplog.records)
    assert all(getattr(r, "error", "") == "SessionNotFound" for r in step_records)


def test_span_exception_event_has_type_only(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "get_tracer", lambda: provider.get_tracer("t"))

    with pytest.raises(ValueError), tracing.start_span("step"):
        raise ValueError("user text: ali@example.com")

    (span,) = exporter.get_finished_spans()
    assert not span.status.is_ok
    events = [e for e in span.events if e.name == "exception"]
    assert events and events[0].attributes["exception.type"] == "ValueError"
    for event in span.events:
        assert "ali@example.com" not in str(dict(event.attributes or {}))
    assert "ali@example.com" not in (span.status.description or "")
