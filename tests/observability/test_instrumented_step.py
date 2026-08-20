"""Tests for `instrumented_step`, the combined logging + metrics context manager."""

from __future__ import annotations

import logging

import pytest

from kuhaku.core.observability import instrumented_step
from kuhaku.core.observability.metrics import STAGE_DURATION
from tests.conftest import prometheus_histogram_count


def _histogram_count(stage: str) -> float:
    return prometheus_histogram_count(STAGE_DURATION, stage=stage)


def test_instrumented_step_logs_and_records_metric(caplog):
    logger = logging.getLogger("test.instrumented_step.ok")
    before = _histogram_count("unit_test_stage_ok")
    with caplog.at_level(logging.INFO, logger="test.instrumented_step.ok"):
        with instrumented_step("unit_test_stage_ok", logger) as rec:
            rec.set(chunk_count=3)

    assert _histogram_count("unit_test_stage_ok") == before + 1
    record = caplog.records[-1]
    assert record.step == "unit_test_stage_ok"
    assert record.status == "ok"
    assert record.chunk_count == 3


def test_instrumented_step_records_metric_and_reraises_on_error(caplog):
    logger = logging.getLogger("test.instrumented_step.error")
    before = _histogram_count("unit_test_stage_error")
    with caplog.at_level(logging.INFO, logger="test.instrumented_step.error"):
        with pytest.raises(RuntimeError):
            with instrumented_step("unit_test_stage_error", logger):
                raise RuntimeError("nope")

    # The histogram observes the duration even on failure — a failing stage should
    # still show up in latency metrics, not disappear from them.
    assert _histogram_count("unit_test_stage_error") == before + 1
    assert caplog.records[-1].status == "error"


def test_instrumented_step_yields_a_recorder_usable_for_extra_fields(caplog):
    logger = logging.getLogger("test.instrumented_step.extra")
    with caplog.at_level(logging.INFO, logger="test.instrumented_step.extra"):
        with instrumented_step("unit_test_stage_extra", logger, initial=1) as rec:
            rec.set(computed=2)
    record = caplog.records[-1]
    assert record.initial == 1
    assert record.computed == 2
