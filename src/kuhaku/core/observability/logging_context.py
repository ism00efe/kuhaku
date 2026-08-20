"""Structured logging primitives: trace-id propagation and JSON formatting.

A single ``trace_id`` is generated once per request and must show up on every log line
emitted while that request is in flight — including lines from nested layers (retriever,
reranker, LLM provider) that have no idea a "request" exists. Rather than threading a
``trace_id`` parameter through every function signature, we bind it to a ``ContextVar``
for the duration of the request and inject it into log records via a ``logging.Filter``
attached to the handler. Every layer keeps its existing signature; every log line still
gets the id.

SECURITY: log messages and the structured ``extra`` fields used here must never carry raw
user-supplied text (queries, log content). Only fixed step names, durations, counts, and
low-cardinality category labels are logged — verified by
``tests/security/test_pii_leak_scan.py``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

# Sentinel for "no trace bound". Rendered into logs as-is so an unbound line is
# visually obvious rather than silently blank.
_UNBOUND = "-"

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default=_UNBOUND)

# Fixed, known set of structured fields a log record may carry. Keeping this explicit
# (rather than dumping the whole record.__dict__) means the JSON schema is stable and
# nothing accidental — like a raw query string passed via `extra=` by mistake — leaks in.
_EXTRA_FIELDS = (
    "step",
    "duration_ms",
    "status",
    "chunk_count",
    "redaction_count",
    "citation_count",
    "candidate_count",
    "result_count",
    "strategy",
    "reason",
    "token_count",
    "prompt_token_count",
    "error",
    # FR3 token tracking. `provider`/`model` are fixed, low-cardinality configured
    # strings, never user content.
    "provider",
    "model",
    # HTTP layer. An int from a fixed set — PII-safe by construction. Note what is
    # deliberately absent: the client IP. IP addresses are one of the six categories
    # `sanitization.py` exists to mask, so the API keys its rate limiter on the address
    # in memory and never logs it.
    "status_code",
    # POST /api/ingest audit trail. Unlike a query or log excerpt, a filename is not
    # end-user free text — it is named by a trusted internal operator uploading a
    # knowledge-base document, and knowing *which* document changed the KB is the whole
    # point of logging this endpoint. Still bytes, not raw content: nothing here can carry
    # the six sanitization.py categories the way a query or log body could. Named
    # `upload_filename`, not `filename`: the latter is a reserved `LogRecord` attribute
    # (the source .py file of the log call) — passing it via `extra` raises `KeyError`.
    "upload_filename",
    "file_size_bytes",
    # FR4 citation verification: the raw `[S#]` tags a response cited that did not match
    # any retrieved source. Not end-user content -- a small, fixed-shape token the model
    # emitted, not text copied from the user's query or a document body.
    "invalid_citations",
    # FR3 feedback: a closed two-value word ("positive"/"negative"), enforced by the
    # `Literal` on the wire schema before this code ever runs -- as safe to log as
    # `status` already is.
    "feedback",
    # Guard v2 (D39). All computed/fixed values -- never raw query or LLM output text.
    "guard_zone",  # closed 3-value word ("pass"/"restricted"/"reject")
    "stage1_score",  # a float the classifier computed, not user content
    "stage2_score",  # a float the classifier computed, or absent when Stage-2 didn't run
    "guard_component",  # closed word set, which subsystem degraded (FR6)
    "escalation_reason",  # closed word set (None/"threshold"/"sampled"/"elevated_access")
    "norm_drift",  # an int (character-count delta), not the text itself
    "canary_detected",  # a bool
    "pii_egress_detected",  # a bool
)


def new_trace_id() -> str:
    """Generate a short, sufficiently-unique trace id."""

    return uuid.uuid4().hex[:12]


def get_trace_id() -> str:
    """Return the trace id bound to the current context, or ``"-"`` if none."""

    return _trace_id_var.get()


def has_trace_id() -> bool:
    """True when a trace id is currently bound."""

    return _trace_id_var.get() != _UNBOUND


@contextmanager
def bind_trace_id(trace_id: str | None = None) -> Iterator[str]:
    """Bind ``trace_id`` for the duration of the block.

    Every log record emitted anywhere during the block — including from nested calls
    that never see ``trace_id`` as a parameter — picks it up via :class:`TraceIdFilter`.

    With no argument, an id already bound by an outer scope is *inherited* rather than
    replaced, and only a genuinely unbound context mints a new one. That matters because
    the layer that opens a request (the HTTP API) and the layer that does the work
    (``RAGEngine.answer``) both bind: without inheritance one request would be split
    across two ids, and the api-level line for a failed request could not be joined to
    the pipeline line that explains the failure. Passing an explicit id always wins, so a
    caller that means to start a distinct trace still can.
    """

    current = _trace_id_var.get()
    tid = trace_id or (current if current != _UNBOUND else new_trace_id())
    token = _trace_id_var.set(tid)
    try:
        yield tid
    finally:
        _trace_id_var.reset(token)


class TraceIdFilter(logging.Filter):
    """Attach the current trace id to every record that doesn't already have one.

    Attached to a *handler* (not a logger) so it applies to every record reaching that
    handler, including from third-party libraries — guaranteeing ``%(trace_id)s`` /
    the JSON formatter never hits a missing attribute.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = get_trace_id()
        return True


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON line.

    Only the fixed base fields plus any of :data:`_EXTRA_FIELDS` present on the record
    are included — see the module docstring for why that allowlist matters.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _StepRecorder:
    """Mutable bag a ``log_step`` caller uses to attach fields discovered mid-block."""

    def __init__(self, initial: dict[str, object]) -> None:
        self.extra: dict[str, object] = dict(initial)

    def set(self, **fields: object) -> None:
        self.extra.update(fields)


@contextmanager
def log_step(
    step: str, logger: logging.Logger | None = None, **initial_extra: object
) -> Iterator[_StepRecorder]:
    """Time a pipeline step and log its outcome as a single structured record.

    Usage::

        with log_step("retrieve") as rec:
            retrieved = engine.retrieve(query)
            rec.set(chunk_count=len(retrieved))

    On success logs at INFO with ``status="ok"``; on exception logs at ERROR with
    ``status="error"`` and the exception message, then re-raises (never swallows).
    """

    log = logger or logging.getLogger("kuhaku.core.observability")
    recorder = _StepRecorder(initial_extra)
    start = time.perf_counter()
    try:
        yield recorder
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.error(
            "step failed: %s",
            step,
            extra={
                "step": step,
                "duration_ms": duration_ms,
                "status": "error",
                "error": str(exc),
                **recorder.extra,
            },
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.info(
            "step completed: %s",
            step,
            extra={
                "step": step,
                "duration_ms": duration_ms,
                "status": "ok",
                **recorder.extra,
            },
        )
