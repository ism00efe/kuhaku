"""ExtractionSettings: every field changes something, and an argument beats it."""

from __future__ import annotations

import json

from kuhaku.tools.extraction import (
    ExtractionSettings,
    ExtractionStatus,
    Extractor,
)

from .conftest import Complaint, Order, ScriptedLLM

GOOD = {"data": {"product": "tea", "quantity": 5}}
BAD = "not json"


def test_max_retries_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("KUHAKU_EXTRACTION__MAX_RETRIES", "0")
    llm = ScriptedLLM(BAD)

    result = Extractor(Order, llm=llm).extract("5 tea")

    assert result.status is ExtractionStatus.FAILED
    assert len(llm.calls) == 1


def test_an_argument_beats_the_setting(monkeypatch):
    monkeypatch.setenv("KUHAKU_EXTRACTION__MAX_RETRIES", "0")
    llm = ScriptedLLM(BAD, GOOD)

    result = Extractor(Order, llm=llm, max_retries=1).extract("5 tea")

    assert result.status is ExtractionStatus.COMPLETE


def test_settings_object_is_used_when_passed():
    llm = ScriptedLLM(BAD)
    settings = ExtractionSettings(max_retries=0)

    Extractor(Order, llm=llm, extraction_settings=settings).extract("5 tea")

    assert len(llm.calls) == 1


def test_audit_log_path_comes_from_the_environment(monkeypatch, tmp_path):
    log = tmp_path / "audit.jsonl"
    monkeypatch.setenv("KUHAKU_EXTRACTION__AUDIT_LOG_PATH", str(log))

    Extractor(Order, llm=ScriptedLLM(GOOD)).extract("5 tea")

    assert len(log.read_text(encoding="utf-8").splitlines()) == 1


def test_the_routing_call_is_not_audited_as_an_extraction(monkeypatch, tmp_path):
    log = tmp_path / "audit.jsonl"
    monkeypatch.setenv("KUHAKU_EXTRACTION__AUDIT_LOG_PATH", str(log))
    llm = ScriptedLLM({"data": {"choice": "Order"}}, GOOD)
    router = Extractor([Extractor(Order, llm=llm), Extractor(Complaint, llm=llm)])

    router.extract("5 tea")

    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["metadata"]["schema"] for r in records] == ["Order"]
