"""OpenTelemetry metrics, exported through the OTel Prometheus exporter.

Collection always happens -- incrementing an in-memory instrument costs microseconds --
so this module has no on/off switch of its own; the Prometheus exposition endpoint that
reads these values is a separate, already-optional concern (the JWT-protected
metrics route an embedding application chooses to expose, and ``metrics_summary.py``'s
own registry read here). ``telemetry.py`` installs a ``PrometheusMetricReader`` that
registers itself into ``prometheus_client``'s default ``REGISTRY`` -- every instrument
below is created against the OTel API, but still ends up served through the exact same
Prometheus exposition path that existed before this module used OTel at all.

SECURITY: every attribute used here is a fixed, low-cardinality category (a stage name, a
sanitization category like "[EMAIL]", a guard zone, a status word) -- never raw request
content. The exposition endpoint therefore cannot leak query text or PII by construction;
``tests/security/test_pii_leak_scan.py`` verifies this holds.
"""

from __future__ import annotations

import logging

from ..sanitization import Redaction
from .telemetry import get_meter
from .tracing import GEN_AI_REQUEST_MODEL, GEN_AI_SYSTEM, GEN_AI_TOKEN_TYPE

logger = logging.getLogger(__name__)

_meter = get_meter()

STAGE_DURATION = _meter.create_histogram(
    "kuhaku_stage_duration_seconds", unit="s", description="Duration of each pipeline stage"
)
REDACTIONS = _meter.create_counter(
    "kuhaku_sanitization_redactions_total",
    unit="1",
    description="PII items masked, by category",
)
LLM_TOKENS_TOTAL = _meter.create_counter(
    "kuhaku_llm_tokens_total",
    unit="{token}",
    description="LLM tokens processed, by provider/model and GenAI token type",
)

# --- Category 2: user feedback -----------------------------------------------------
FEEDBACK_TOTAL = _meter.create_counter(
    "kuhaku_feedback_total", unit="1", description="User feedback submissions, by value"
)

# Transport-layer metrics live in their own `api_*` namespace rather than extending
# `rag_*`: `rag_requests_total{status}` counts answers the *engine* produced, so reusing
# it for HTTP would double-count the same request at two different layers. `endpoint` is
# always a fixed route template (never a raw request path, which would be unbounded
# cardinality and could carry user content); `outcome` is a fixed word set.
API_REQUESTS = _meter.create_counter(
    "api_requests_total", unit="1", description="HTTP API requests, by endpoint and outcome"
)
API_REQUEST_DURATION = _meter.create_histogram(
    "api_request_duration_seconds", unit="s", description="HTTP API request duration"
)

def record_redactions(redactions: list[Redaction]) -> None:
    """Record sanitization counts, one increment per masked category (never the values)."""

    for r in redactions:
        REDACTIONS.add(r.count, {"category": r.label})


# --- Prompt Injection Guard v2 (normalize -> two-stage classify -> 3-zone) -----------
# Generic, tool-agnostic input-validation infrastructure (core.security.guard/classifier)
# -- these metrics used to carry a misleading "rag_" prefix even though guard.py itself
# has nothing RAG-specific about it. The 3 output-side ones (canary/PII egress/ungrounded
# citations), by contrast, ARE RAG-scoped -- only tools/rag/engine.py emits them -- so
# they live in tools/rag/metrics.py instead, not here.
GUARD_STAGE1_ESCALATIONS = _meter.create_counter(
    "kuhaku_guard_stage1_escalations_total",
    unit="1",
    description="Requests escalated from Stage-1 to Stage-2 classification, by reason",
)
GUARD_STAGE2_CLASSIFICATIONS = _meter.create_counter(
    "kuhaku_guard_stage2_classifications_total",
    unit="1",
    description="Stage-2 classification outcomes, by result",
)
GUARD_ZONE = _meter.create_counter(
    "kuhaku_guard_zone_total", unit="1", description="Guard v2 decisions, by zone"
)
GUARD_DEGRADATION = _meter.create_counter(
    "kuhaku_guard_degradation_total",
    unit="1",
    description=(
        "Guard v2 component degradation events (fell back to a safer default), by component"
    ),
)

_VALID_GUARD_ZONES = {"pass", "restricted", "reject"}
_VALID_ESCALATION_REASONS = {"threshold", "sampled", "elevated_access"}
_VALID_STAGE2_RESULTS = {"safe", "unsafe"}
_VALID_GUARD_COMPONENTS = {"stage1", "stage2", "tokenizer", "audit", "output_guard"}


def record_guard_zone(zone: str) -> None:
    """Increment the guard-v2 zone counter. Clamped: `zone` is always one of the
    3 fixed values the guard pipeline itself produces, but the clamp is kept as a
    defensive floor against a future bug upstream, not an expected path."""

    label = zone if zone in _VALID_GUARD_ZONES else "unknown"
    GUARD_ZONE.add(1, {"zone": label})


def record_guard_stage1_escalation(reason: str) -> None:
    """Increment the Stage-1 -> Stage-2 escalation counter, by reason."""

    label = reason if reason in _VALID_ESCALATION_REASONS else "unknown"
    GUARD_STAGE1_ESCALATIONS.add(1, {"reason": label})


def record_guard_stage2_classification(result: str) -> None:
    """Increment the Stage-2 classification-outcome counter, by result."""

    label = result if result in _VALID_STAGE2_RESULTS else "unknown"
    GUARD_STAGE2_CLASSIFICATIONS.add(1, {"result": label})


def record_guard_degradation(component: str) -> None:
    """Increment the guard-v2 degradation counter, by component."""

    label = component if component in _VALID_GUARD_COMPONENTS else "unknown"
    GUARD_DEGRADATION.add(1, {"component": label})


# --- Retry: LLM, embedding, vector store, reranker call sites ------------------------
RETRY_ATTEMPTS = _meter.create_counter(
    "kuhaku_retry_attempts_total", unit="1", description="Retry attempts by service"
)
RETRY_SUCCESSES = _meter.create_counter(
    "kuhaku_retry_successes_total", unit="1", description="Successful retries by service"
)
RETRY_FAILURES = _meter.create_counter(
    "kuhaku_retry_failures_total",
    unit="1",
    description="Failed retries (all attempts exhausted) by service",
)

_VALID_RETRY_SERVICES = {"llm", "embedding", "vectorstore", "reranker"}


def record_retry_attempt(service: str) -> None:
    """Increment the retry-attempts counter, by service."""

    label = service if service in _VALID_RETRY_SERVICES else "unknown"
    RETRY_ATTEMPTS.add(1, {"service": label})


def record_retry_success(service: str) -> None:
    """Increment the retry-successes counter, by service.

    Only called when a call succeeded after at least one retry -- a first-try success
    is not a "retry success" and does not touch this counter.
    """

    label = service if service in _VALID_RETRY_SERVICES else "unknown"
    RETRY_SUCCESSES.add(1, {"service": label})


def record_retry_failure(service: str) -> None:
    """Increment the retry-failures counter, by service: all attempts exhausted."""

    label = service if service in _VALID_RETRY_SERVICES else "unknown"
    RETRY_FAILURES.add(1, {"service": label})


# --- Authentication & audit ----------------------------------------------------------
AUTH_LOGIN_TOTAL = _meter.create_counter(
    "kuhaku_auth_login_total", unit="1", description="Login attempts, by outcome"
)
AUTH_REFRESH_TOTAL = _meter.create_counter(
    "kuhaku_auth_refresh_total",
    unit="1",
    description="Refresh-token rotations (successful token-refresh calls)",
)
AUTH_LOGOUT_TOTAL = _meter.create_counter(
    "kuhaku_auth_logout_total", unit="1", description="Logout calls (refresh-token revocations)"
)
AUDIT_RECORDS_TOTAL = _meter.create_counter(
    "kuhaku_audit_records_total",
    unit="1",
    description="Audit records written, across all call sites",
)

_VALID_AUTH_LOGIN_STATUSES = {"success", "failure"}


def record_auth_login(status: str) -> None:
    """Increment the login-attempts counter, by outcome."""

    label = status if status in _VALID_AUTH_LOGIN_STATUSES else "unknown"
    AUTH_LOGIN_TOTAL.add(1, {"status": label})


# --- Evaluation metrics infrastructure ------------------------------------------------
# EVALUATION_RUN_COUNT/DURATION renamed rag_* -> framework_* (later kuhaku_* when
# the package itself was renamed): the benchmark harness
# (kuhaku/evaluation/) is generic -- it runs over any `EvaluationTarget`, not just RAG
# (see kuhaku/__init__.py's module docstring) -- and "a benchmark run completed"/"run
# duration" carry no RAG-specific vocabulary, unlike the grounded-generation-specific
# faithfulness/hallucination gauges, which live in tools/rag/metrics.py instead.
EVALUATION_RUN_COUNT = _meter.create_counter(
    "kuhaku_evaluation_run_count_total",
    unit="1",
    description="Benchmark evaluation runs completed, by final status",
)
EVALUATION_RUN_DURATION = _meter.create_histogram(
    "kuhaku_evaluation_run_duration_seconds",
    unit="s",
    description="Wall-clock duration of one benchmark evaluation run",
)

_VALID_EVALUATION_RUN_STATUSES = {"completed", "failed"}


def record_evaluation_run(status: str, duration_seconds: float) -> None:
    """Increment the evaluation-run counter and observe its duration."""

    label = status if status in _VALID_EVALUATION_RUN_STATUSES else "unknown"
    EVALUATION_RUN_COUNT.add(1, {"status": label})
    EVALUATION_RUN_DURATION.record(duration_seconds)


# --- LLM token usage, GenAI semantic conventions (Feature 4) ---------------------------
# Same Prometheus series name/shape as before this migration (one Counter, incremented
# once per token type per generate() call) -- only the attribute keys/values changed, to
# the GenAI convention: `gen_ai.system` (was `provider`), `gen_ai.request.model` (new),
# `gen_ai.token.type` with values "input"/"output" (was `type` with "prompt"/"completion").
def record_llm_token_usage(
    *, provider: str, model: str, input_tokens: int, output_tokens: int
) -> None:
    """Increment the LLM token-usage counter for one ``generate()`` call.

    Called from ``core.llm.token_tracking.TokenTrackingLLM``, the single call site that
    already extracts provider/model/token counts for its log line.
    """

    LLM_TOKENS_TOTAL.add(
        input_tokens,
        {GEN_AI_SYSTEM: provider, GEN_AI_REQUEST_MODEL: model, GEN_AI_TOKEN_TYPE: "input"},
    )
    LLM_TOKENS_TOTAL.add(
        output_tokens,
        {GEN_AI_SYSTEM: provider, GEN_AI_REQUEST_MODEL: model, GEN_AI_TOKEN_TYPE: "output"},
    )
