"""Tests for OTel metrics collection and the (mocked) exposition server.

RAG-specific metrics (retrieval, cache, citations, faithfulness, ...) are tested in
tests/rag/test_metrics.py instead -- see tools/rag/metrics.py's module docstring.
"""

from __future__ import annotations

from kuhaku.core.observability.metrics import (
    AUDIT_RECORDS_TOTAL,
    AUTH_LOGIN_TOTAL,
    AUTH_LOGOUT_TOTAL,
    AUTH_REFRESH_TOTAL,
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
from kuhaku.core.observability.tracing import GEN_AI_SYSTEM, GEN_AI_TOKEN_TYPE
from kuhaku.core.sanitization import Redaction
from tests.conftest import (
    prometheus_counter_value as _counter_value,
)
from tests.conftest import (
    prometheus_histogram_count as _histogram_count,
)


# --- counters/histograms are wired correctly --------------------------------
def test_stage_duration_observes():
    before = _histogram_count(STAGE_DURATION, stage="sanitize")
    STAGE_DURATION.record(0.01, {"stage": "sanitize"})
    assert _histogram_count(STAGE_DURATION, stage="sanitize") == before + 1


# --- record_redactions --------------------------------------------------------
def test_record_redactions_increments_by_category():
    before = _counter_value(REDACTIONS, category="[EMAIL]")
    record_redactions([Redaction("[EMAIL]", 2), Redaction("[CARD]", 1)])
    assert _counter_value(REDACTIONS, category="[EMAIL]") == before + 2


def test_record_redactions_empty_list_is_noop():
    before = _counter_value(REDACTIONS, category="[IP]")
    record_redactions([])
    assert _counter_value(REDACTIONS, category="[IP]") == before


def test_record_llm_token_usage_uses_gen_ai_attribute_keys():
    """D57: token usage carries GenAI semantic-convention attribute keys
    (gen_ai.system, gen_ai.token.type="input"/"output") instead of the old
    provider/type=prompt|completion pair -- same Prometheus series name/shape."""

    before_input = _counter_value(
        LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "ollama", GEN_AI_TOKEN_TYPE: "input"}
    )
    before_output = _counter_value(
        LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "ollama", GEN_AI_TOKEN_TYPE: "output"}
    )
    record_llm_token_usage(provider="ollama", model="m", input_tokens=42, output_tokens=7)
    assert (
        _counter_value(LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "ollama", GEN_AI_TOKEN_TYPE: "input"})
        == before_input + 42
    )
    assert (
        _counter_value(LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "ollama", GEN_AI_TOKEN_TYPE: "output"})
        == before_output + 7
    )


def test_feedback_total_counter_by_value():
    before = _counter_value(FEEDBACK_TOTAL, feedback="positive")
    FEEDBACK_TOTAL.add(1, {"feedback": "positive"})
    assert _counter_value(FEEDBACK_TOTAL, feedback="positive") == before + 1


# --- Prompt Injection Guard v2 (D39) ------------------------------------------
def test_record_guard_zone_uses_the_zone_as_the_label():
    before = _counter_value(GUARD_ZONE, zone="restricted")
    record_guard_zone("restricted")
    assert _counter_value(GUARD_ZONE, zone="restricted") == before + 1


def test_record_guard_zone_clamps_unknown_value():
    before = _counter_value(GUARD_ZONE, zone="unknown")
    record_guard_zone("something-unexpected")
    assert _counter_value(GUARD_ZONE, zone="unknown") == before + 1


def test_record_guard_stage1_escalation_uses_the_reason_as_the_label():
    before = _counter_value(GUARD_STAGE1_ESCALATIONS, reason="sampled")
    record_guard_stage1_escalation("sampled")
    assert _counter_value(GUARD_STAGE1_ESCALATIONS, reason="sampled") == before + 1


def test_record_guard_stage1_escalation_clamps_unknown_value():
    before = _counter_value(GUARD_STAGE1_ESCALATIONS, reason="unknown")
    record_guard_stage1_escalation("bogus")
    assert _counter_value(GUARD_STAGE1_ESCALATIONS, reason="unknown") == before + 1


def test_record_guard_stage2_classification_uses_the_result_as_the_label():
    before = _counter_value(GUARD_STAGE2_CLASSIFICATIONS, result="unsafe")
    record_guard_stage2_classification("unsafe")
    assert _counter_value(GUARD_STAGE2_CLASSIFICATIONS, result="unsafe") == before + 1


def test_record_guard_stage2_classification_clamps_unknown_value():
    before = _counter_value(GUARD_STAGE2_CLASSIFICATIONS, result="unknown")
    record_guard_stage2_classification("maybe")
    assert _counter_value(GUARD_STAGE2_CLASSIFICATIONS, result="unknown") == before + 1


def test_record_guard_degradation_uses_the_component_as_the_label():
    before = _counter_value(GUARD_DEGRADATION, component="stage2")
    record_guard_degradation("stage2")
    assert _counter_value(GUARD_DEGRADATION, component="stage2") == before + 1


def test_record_guard_degradation_clamps_unknown_value():
    before = _counter_value(GUARD_DEGRADATION, component="unknown")
    record_guard_degradation("bogus")
    assert _counter_value(GUARD_DEGRADATION, component="unknown") == before + 1


# --- Retry (D40) ---------------------------------------------------------------
def test_record_retry_attempt_uses_the_service_as_the_label():
    before = _counter_value(RETRY_ATTEMPTS, service="llm")
    record_retry_attempt("llm")
    assert _counter_value(RETRY_ATTEMPTS, service="llm") == before + 1


def test_record_retry_attempt_clamps_unknown_value():
    before = _counter_value(RETRY_ATTEMPTS, service="unknown")
    record_retry_attempt("bogus")
    assert _counter_value(RETRY_ATTEMPTS, service="unknown") == before + 1


def test_record_retry_success_uses_the_service_as_the_label():
    before = _counter_value(RETRY_SUCCESSES, service="embedding")
    record_retry_success("embedding")
    assert _counter_value(RETRY_SUCCESSES, service="embedding") == before + 1


def test_record_retry_success_clamps_unknown_value():
    before = _counter_value(RETRY_SUCCESSES, service="unknown")
    record_retry_success("bogus")
    assert _counter_value(RETRY_SUCCESSES, service="unknown") == before + 1


def test_record_retry_failure_uses_the_service_as_the_label():
    before = _counter_value(RETRY_FAILURES, service="vectorstore")
    record_retry_failure("vectorstore")
    assert _counter_value(RETRY_FAILURES, service="vectorstore") == before + 1


def test_record_retry_failure_clamps_unknown_value():
    before = _counter_value(RETRY_FAILURES, service="unknown")
    record_retry_failure("bogus")
    assert _counter_value(RETRY_FAILURES, service="unknown") == before + 1

# --- Authentication & audit (D41) --------------------------------------------------
def test_record_auth_login_success_uses_the_status_as_the_label():
    before = _counter_value(AUTH_LOGIN_TOTAL, status="success")
    record_auth_login("success")
    assert _counter_value(AUTH_LOGIN_TOTAL, status="success") == before + 1


def test_record_auth_login_failure_uses_the_status_as_the_label():
    before = _counter_value(AUTH_LOGIN_TOTAL, status="failure")
    record_auth_login("failure")
    assert _counter_value(AUTH_LOGIN_TOTAL, status="failure") == before + 1


def test_record_auth_login_clamps_unknown_status():
    before = _counter_value(AUTH_LOGIN_TOTAL, status="unknown")
    record_auth_login("bogus")
    assert _counter_value(AUTH_LOGIN_TOTAL, status="unknown") == before + 1


def test_auth_refresh_total_counter_increments():
    before = _counter_value(AUTH_REFRESH_TOTAL)
    AUTH_REFRESH_TOTAL.add(1)
    assert _counter_value(AUTH_REFRESH_TOTAL) == before + 1


def test_auth_logout_total_counter_increments():
    before = _counter_value(AUTH_LOGOUT_TOTAL)
    AUTH_LOGOUT_TOTAL.add(1)
    assert _counter_value(AUTH_LOGOUT_TOTAL) == before + 1


def test_audit_records_total_counter_increments():
    before = _counter_value(AUDIT_RECORDS_TOTAL)
    AUDIT_RECORDS_TOTAL.add(1)
    assert _counter_value(AUDIT_RECORDS_TOTAL) == before + 1
