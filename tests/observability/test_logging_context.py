"""Tests for trace-id propagation and structured (JSON) logging."""

from __future__ import annotations

import json
import logging

import pytest

from kuhaku.core.observability.logging_context import (
    JsonFormatter,
    TraceIdFilter,
    bind_trace_id,
    get_trace_id,
    log_step,
    new_trace_id,
)


def _make_record(msg: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# --- trace id lifecycle ------------------------------------------------------
def test_get_trace_id_default_is_dash():
    assert get_trace_id() == "-"


def test_new_trace_id_is_unique_and_short():
    a, b = new_trace_id(), new_trace_id()
    assert a != b
    assert len(a) == 12


def test_bind_trace_id_generates_when_not_given():
    with bind_trace_id() as tid:
        assert tid == get_trace_id()
        assert tid != "-"


def test_bind_trace_id_accepts_explicit_id():
    with bind_trace_id("abc123") as tid:
        assert tid == "abc123"
        assert get_trace_id() == "abc123"


def test_bind_trace_id_inherits_an_outer_id_when_not_given():
    """A nested bind must not split one request across two ids.

    The HTTP API binds an id per request and RAGEngine.answer() binds again inside it;
    if the inner call minted a fresh id, the api-level and pipeline log lines for the
    same request would no longer join.
    """

    with bind_trace_id("outer"):
        with bind_trace_id() as inner:
            assert inner == "outer"
            assert get_trace_id() == "outer"


def test_bind_trace_id_restores_previous_on_exit():
    with bind_trace_id("outer"):
        with bind_trace_id("inner"):
            assert get_trace_id() == "inner"
        assert get_trace_id() == "outer"
    assert get_trace_id() == "-"


def test_bind_trace_id_restores_on_exception():
    with pytest.raises(ValueError):
        with bind_trace_id("will-reset"):
            raise ValueError("boom")
    assert get_trace_id() == "-"


# --- TraceIdFilter -------------------------------------------------------------
def test_filter_injects_current_trace_id():
    with bind_trace_id("filter-test"):
        record = _make_record()
        assert TraceIdFilter().filter(record) is True
        assert record.trace_id == "filter-test"


def test_filter_does_not_override_existing_trace_id():
    record = _make_record(trace_id="pre-set")
    TraceIdFilter().filter(record)
    assert record.trace_id == "pre-set"


def test_filter_defaults_to_dash_outside_any_bound_context():
    record = _make_record()
    TraceIdFilter().filter(record)
    assert record.trace_id == "-"


# --- JsonFormatter ---------------------------------------------------------
def test_json_formatter_produces_valid_json_with_base_fields():
    record = _make_record("hi there", trace_id="t1")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hi there"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["trace_id"] == "t1"
    assert "timestamp" in payload


def test_json_formatter_includes_known_extra_fields_only_when_present():
    record = _make_record(step="retrieve", duration_ms=12.5, status="ok")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["step"] == "retrieve"
    assert payload["duration_ms"] == 12.5
    assert payload["status"] == "ok"
    # Fields never set must simply be absent, not null/empty.
    assert "chunk_count" not in payload
    assert "error" not in payload


def test_json_formatter_ignores_unknown_extra_attributes():
    """Only the fixed allowlist is emitted — an unexpected attribute (e.g. accidentally
    passing raw query text via `extra=`) must never leak into the JSON output."""
    record = _make_record()
    record.totally_unexpected_field = "raw user query with card 4111111111111111"
    payload = json.loads(JsonFormatter().format(record))
    assert "totally_unexpected_field" not in json.dumps(payload)
    assert "4111111111111111" not in json.dumps(payload)


def test_json_formatter_missing_trace_id_defaults_to_dash():
    record = _make_record()
    payload = json.loads(JsonFormatter().format(record))
    assert payload["trace_id"] == "-"


def test_json_formatter_includes_exception_info():
    try:
        raise RuntimeError("kaboom")
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            name="t", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="failed", args=(), exc_info=sys.exc_info(),
        )
    payload = json.loads(JsonFormatter().format(record))
    assert "kaboom" in payload["exc_info"]


# --- log_step ----------------------------------------------------------------
def test_log_step_logs_success(caplog):
    logger = logging.getLogger("test.log_step.success")
    with caplog.at_level(logging.INFO, logger="test.log_step.success"):
        with log_step("sanitize", logger) as rec:
            rec.set(redaction_count=2)
    record = caplog.records[-1]
    assert record.step == "sanitize"
    assert record.status == "ok"
    assert record.redaction_count == 2
    assert record.duration_ms >= 0


def test_log_step_logs_error_and_reraises(caplog):
    logger = logging.getLogger("test.log_step.error")
    with caplog.at_level(logging.INFO, logger="test.log_step.error"):
        with pytest.raises(ValueError, match="boom"):
            with log_step("generate", logger):
                raise ValueError("boom")
    record = caplog.records[-1]
    assert record.step == "generate"
    assert record.status == "error"
    assert record.error == "boom"


def test_log_step_initial_extra_is_included(caplog):
    logger = logging.getLogger("test.log_step.initial")
    with caplog.at_level(logging.INFO, logger="test.log_step.initial"):
        with log_step("rerank", logger, candidate_count=20):
            pass
    record = caplog.records[-1]
    assert record.candidate_count == 20


def test_log_step_uses_default_logger_when_none_given(caplog):
    with caplog.at_level(logging.INFO, logger="kuhaku.core.observability"):
        with log_step("sanitize"):
            pass
    assert any(r.step == "sanitize" for r in caplog.records)
