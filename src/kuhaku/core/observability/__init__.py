"""Observability: structured logging with trace-id propagation, OTel metrics, OTel tracing.

``instrumented_step`` is the one thing most callers need: it opens an OTel span, times a
pipeline stage, logs a structured record for it (via :mod:`.logging_context`), and records
the same duration in the ``STAGE_DURATION`` histogram (via :mod:`.metrics`) — so
instrumenting a stage is a single ``with`` block instead of three.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from .logging_context import (
    JsonFormatter,
    TraceIdFilter,
    bind_trace_id,
    get_trace_id,
    has_trace_id,
    log_step,
    new_trace_id,
)
from .metrics import (
    API_REQUEST_DURATION,
    API_REQUESTS,
    AUDIT_RECORDS_TOTAL,
    AUTH_LOGIN_TOTAL,
    AUTH_LOGOUT_TOTAL,
    AUTH_REFRESH_TOTAL,
    EVALUATION_RUN_COUNT,
    EVALUATION_RUN_DURATION,
    FEEDBACK_TOTAL,
    GUARD_DEGRADATION,
    GUARD_STAGE1_ESCALATIONS,
    GUARD_STAGE2_CLASSIFICATIONS,
    GUARD_ZONE,
    LLM_TOKENS_TOTAL,
    REDACTIONS,
    RETRY_ATTEMPTS,
    RETRY_FAILURES,
    RETRY_SUCCESSES,
    STAGE_DURATION,
    record_auth_login,
    record_evaluation_run,
    record_guard_degradation,
    record_guard_stage1_escalation,
    record_guard_stage2_classification,
    record_guard_zone,
    record_llm_token_usage,
    record_redactions,
    record_retry_attempt,
    record_retry_failure,
    record_retry_success,
)
from .metrics_summary import build_metrics_summary, get_cached_metrics_summary
from .telemetry import get_meter, get_tracer
from .tracing import set_current_span_attributes, set_span_attributes, start_span


class _InstrumentedStepRecorder:
    """Bridges ``log_step``'s ``_StepRecorder`` with the current OTel span.

    ``rec.set(...)`` -- already used at every existing ``instrumented_step`` call site --
    now also updates the open span's attributes, at zero cost to those call sites.
    """

    def __init__(self, log_recorder, span) -> None:  # noqa: ANN001 (internal, structural)
        self._log_recorder = log_recorder
        self._span = span

    def set(self, **fields: object) -> None:
        self._log_recorder.set(**fields)
        set_span_attributes(self._span, fields)


@contextmanager
def instrumented_step(step: str, logger: logging.Logger | None = None, **initial_extra: object):
    """Span + time + log + record-a-metric for one named pipeline stage.

    Usage::

        with instrumented_step("retrieve") as rec:
            retrieved = engine.retrieve(query)
            rec.set(chunk_count=len(retrieved))

    The duration is recorded on ``STAGE_DURATION`` on both the success and the exception
    path (a failing stage must still show up in latency metrics); the span itself gets its
    exception recorded and status set to ERROR by OTel's own ``start_as_current_span``
    (see :func:`.tracing.start_span`), and re-raises unchanged.
    """

    start = time.perf_counter()
    try:
        with start_span(step, **initial_extra) as span:
            with log_step(step, logger, **initial_extra) as log_recorder:
                yield _InstrumentedStepRecorder(log_recorder, span)
    finally:
        STAGE_DURATION.record(time.perf_counter() - start, {"stage": step})


__all__ = [
    "JsonFormatter",
    "TraceIdFilter",
    "bind_trace_id",
    "get_trace_id",
    "has_trace_id",
    "new_trace_id",
    "log_step",
    "instrumented_step",
    # OpenTelemetry tracing
    "get_tracer",
    "get_meter",
    "start_span",
    "set_span_attributes",
    "set_current_span_attributes",
    "STAGE_DURATION",
    "REDACTIONS",
    "LLM_TOKENS_TOTAL",
    "record_llm_token_usage",
    "API_REQUESTS",
    "API_REQUEST_DURATION",
    "record_redactions",
    "FEEDBACK_TOTAL",
    # Prompt Injection Guard v2
    "GUARD_STAGE1_ESCALATIONS",
    "GUARD_STAGE2_CLASSIFICATIONS",
    "GUARD_ZONE",
    "GUARD_DEGRADATION",
    "record_guard_zone",
    "record_guard_stage1_escalation",
    "record_guard_stage2_classification",
    "record_guard_degradation",
    # Retry
    "RETRY_ATTEMPTS",
    "RETRY_SUCCESSES",
    "RETRY_FAILURES",
    "record_retry_attempt",
    "record_retry_success",
    "record_retry_failure",
    # Authentication & audit
    "AUTH_LOGIN_TOTAL",
    "AUTH_REFRESH_TOTAL",
    "AUTH_LOGOUT_TOTAL",
    "AUDIT_RECORDS_TOTAL",
    "record_auth_login",
    # Admin panel metrics summary
    "build_metrics_summary",
    "get_cached_metrics_summary",
    # Evaluation metrics infrastructure
    "EVALUATION_RUN_COUNT",
    "EVALUATION_RUN_DURATION",
    "record_evaluation_run",
]
