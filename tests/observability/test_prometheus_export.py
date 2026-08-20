"""Integration tests: OTel metrics recorded through the public API still reach the real
Prometheus exposition text format (D57).

Calls ``prometheus_client.generate_latest()`` directly with no mocking -- the exact call
``payment_assistant/api/admin_routes.py``'s ``GET /api/admin/metrics`` route makes -- so a
regression here would mean that route breaks too, even though this repo can't import that
route (it lives outside ``kuhaku/``). RAG-specific exposition checks (``rag_*`` series)
live in ``tests/rag/test_metrics.py`` instead -- see ``tools/rag/metrics.py``'s docstring.
"""

from __future__ import annotations

import prometheus_client

from kuhaku.core.observability.metrics import (
    AUTH_LOGIN_TOTAL,
    STAGE_DURATION,
    record_llm_token_usage,
)
from kuhaku.core.observability.tracing import GEN_AI_SYSTEM, GEN_AI_TOKEN_TYPE


def _exposition_text() -> str:
    return prometheus_client.generate_latest().decode("utf-8")


def test_histogram_appears_with_bucket_count_and_sum():
    STAGE_DURATION.record(0.02, {"stage": "unit_test_export_stage"})
    text = _exposition_text()
    assert "kuhaku_stage_duration_seconds_bucket{" in text
    assert "kuhaku_stage_duration_seconds_count{" in text
    assert "kuhaku_stage_duration_seconds_sum{" in text
    assert 'stage="unit_test_export_stage"' in text


def test_duration_histogram_uses_second_scale_buckets_not_millisecond_scale():
    """D57: the SDK's own default histogram boundaries (0, 5, 10, ... 10000) would
    collapse every real duration kuhaku records into the first bucket -- the
    View configured in telemetry.py must reproduce prometheus_client's actual
    second-scale defaults (.005 ... 10.0) instead."""

    STAGE_DURATION.record(0.02, {"stage": "unit_test_export_buckets"})
    bucket_lines = [
        line
        for line in _exposition_text().splitlines()
        if line.startswith("kuhaku_stage_duration_seconds_bucket{")
    ]
    boundaries = {line.split('le="')[1].split('"')[0] for line in bucket_lines}
    assert {"0.005", "0.5", "10.0"} <= boundaries
    assert "10000.0" not in boundaries  # the SDK's millisecond-scale default, unused here


def test_llm_token_usage_exports_sanitized_gen_ai_labels():
    """Prometheus label names can't contain '.': gen_ai.system/gen_ai.token.type export
    as gen_ai_system/gen_ai_token_type (verified against the installed exporter)."""

    record_llm_token_usage(
        provider="unit_test_provider", model="unit_test_model", input_tokens=5, output_tokens=9
    )
    text = _exposition_text()
    assert "kuhaku_llm_tokens_total{" in text
    assert 'gen_ai_system="unit_test_provider"' in text
    assert 'gen_ai_token_type="input"' in text
    assert 'gen_ai_token_type="output"' in text
    assert GEN_AI_SYSTEM not in text  # the dotted form never appears
    assert GEN_AI_TOKEN_TYPE not in text


def test_record_auth_login_reaches_the_exposition_endpoint():
    AUTH_LOGIN_TOTAL.add(1, {"status": "success"})
    text = _exposition_text()
    assert "kuhaku_auth_login_total{" in text
    assert 'status="success"' in text
