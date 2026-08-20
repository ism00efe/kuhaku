"""Tests for the admin metrics summary (D44) and its Prometheus registry read (D57).

``metrics_summary.py`` no longer imports ``prometheus_client`` itself -- it reads the
registry through ``telemetry.collect_prometheus_families``, kuhaku's one
deliberate direct dependency on that package. These tests exercise both the wrapper and
the summary built on top of it.
"""

from __future__ import annotations

from kuhaku.core.observability.metrics import API_REQUESTS
from kuhaku.core.observability.metrics_summary import (
    build_metrics_summary,
    get_cached_metrics_summary,
)
from kuhaku.core.observability.telemetry import collect_prometheus_families


def test_collect_prometheus_families_returns_registered_families():
    API_REQUESTS.add(1, {"endpoint": "/unit-test", "outcome": "ok"})
    families = collect_prometheus_families()
    assert any(family.name == "api_requests" for family in families)


def test_build_metrics_summary_reflects_api_request_count():
    before = build_metrics_summary()["request_count"]
    API_REQUESTS.add(1, {"endpoint": "/unit-test", "outcome": "ok"})
    after = build_metrics_summary()["request_count"]
    assert after == before + 1


def test_build_metrics_summary_never_raises_and_has_expected_keys():
    summary = build_metrics_summary()
    assert set(summary) == {
        "request_count", "error_count", "error_rate",
        "cache_hits", "cache_misses", "cache_hit_ratio",
        "guard_reject_count", "guard_total_count", "guard_reject_rate",
        "avg_response_time_seconds",
    }


def test_get_cached_metrics_summary_adds_a_generated_at_timestamp():
    summary = get_cached_metrics_summary()
    assert "generated_at" in summary
