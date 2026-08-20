"""Tests for `kuhaku.core.security.audit.record`'s `enabled` gate and default-path
fallback (Feature 1, D53/D55).

`path=None`/`""` used to be a silent no-op (audit logging effectively disabled by
omission). It now falls back to `./logs/kuhaku_audit.jsonl`, created relative to the
process's current working directory. Audit logging is security-critical: while `enabled`
(the default), a write failure is never swallowed -- it raises `AuditWriteError`. An
operator who wants no audit logging must say so explicitly via `enabled=False`, which
makes `record()` an immediate, filesystem-untouched no-op.
"""

from __future__ import annotations

import json
import logging

import pytest

from kuhaku.core.security import audit


def _record(path, **overrides) -> None:
    kwargs = dict(trace_id="t1", raw_question="soru?", sanitized_retrieval_query="soru")
    kwargs.update(overrides)
    audit.record(path, **kwargs)


def test_no_path_writes_to_default_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    _record(None)

    default_log = tmp_path / "logs" / "kuhaku_audit.jsonl"
    assert default_log.exists()
    lines = default_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace_id"] == "t1"


def test_empty_string_path_also_uses_default_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    _record("")

    assert (tmp_path / "logs" / "kuhaku_audit.jsonl").exists()


def test_default_logs_directory_created_if_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "logs").exists()

    _record(None)

    assert (tmp_path / "logs").is_dir()


def test_default_logs_directory_reused_if_already_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    _record(None)

    default_log = tmp_path / "logs" / "kuhaku_audit.jsonl"
    assert default_log.exists()


def test_explicit_path_still_used_when_given(tmp_path):
    explicit = tmp_path / "custom" / "audit.jsonl"

    _record(str(explicit))

    assert explicit.exists()
    assert not (tmp_path / "logs").exists()


# --- enabled=False: explicit opt-out, no filesystem interaction ------------------------
def test_disabled_is_a_filesystem_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    _record(None, enabled=False)

    assert not (tmp_path / "logs").exists()


def test_disabled_ignores_an_explicit_path_too(tmp_path):
    explicit = tmp_path / "custom" / "audit.jsonl"

    _record(str(explicit), enabled=False)

    assert not explicit.exists()


def test_enabled_defaults_to_true(tmp_path):
    """Every pre-existing call site never passed `enabled` -- it must keep writing."""

    explicit = tmp_path / "audit.jsonl"

    _record(str(explicit))

    assert explicit.exists()


# --- write failures: never swallowed while enabled (D53/D55) ---------------------------
def test_default_path_unwritable_raises_and_logs(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)

    def _raise_mkdir(self, *args, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr("pathlib.Path.mkdir", _raise_mkdir)

    with caplog.at_level(logging.ERROR, logger="kuhaku.core.security.audit"):
        with pytest.raises(audit.AuditWriteError, match="kuhaku_audit.jsonl"):
            _record(None)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1


def test_explicit_path_unwritable_raises(tmp_path):
    # A directory, not a file, at the target path -- opening it for append raises.
    bad_path = tmp_path / "audit_dir"
    bad_path.mkdir()

    with pytest.raises(audit.AuditWriteError):
        _record(bad_path)


def test_write_failure_does_not_increment_audit_records_total(tmp_path):
    from kuhaku.core.observability.metrics import AUDIT_RECORDS_TOTAL
    from tests.conftest import prometheus_counter_value

    bad_path = tmp_path / "audit_dir"
    bad_path.mkdir()
    before = prometheus_counter_value(AUDIT_RECORDS_TOTAL)
    with pytest.raises(audit.AuditWriteError):
        _record(bad_path)
    assert prometheus_counter_value(AUDIT_RECORDS_TOTAL) == before


# --- read_audit_log / read_recent (D44) ------------------------------------------------
def test_read_audit_log_returns_empty_list_for_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.jsonl"
    assert audit.read_audit_log(str(missing_file)) == []
    assert audit.read_recent(str(missing_file)) == []


def test_read_audit_log_returns_empty_list_and_logs_on_os_error(tmp_path, monkeypatch, caplog):
    log_file = tmp_path / "audit.jsonl"
    log_file.write_text('{"trace_id": "t1"}\n', encoding="utf-8")

    def _raise_read_text(self, *args, **kwargs):
        raise OSError("disk read error")

    monkeypatch.setattr("pathlib.Path.read_text", _raise_read_text)

    with caplog.at_level(logging.ERROR, logger="kuhaku.core.security.audit"):
        records = audit.read_audit_log(str(log_file))

    assert records == []
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "failed to read audit log" in errors[0].message


def test_read_audit_log_returns_records_in_file_order(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    rec1 = {"trace_id": "t1", "event_type": "query"}
    rec2 = {"trace_id": "t2", "event_type": "admin"}
    log_file.write_text(
        f"{json.dumps(rec1)}\n{json.dumps(rec2)}\n",
        encoding="utf-8",
    )

    records = audit.read_audit_log(str(log_file))
    assert len(records) == 2
    assert records[0]["trace_id"] == "t1"
    assert records[1]["trace_id"] == "t2"


def test_read_audit_log_skips_unparsable_lines(tmp_path, caplog):
    log_file = tmp_path / "audit.jsonl"
    log_file.write_text(
        '{"trace_id": "t1"}\nnot-json\n{"trace_id": "t2"}\n',
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="kuhaku.core.security.audit"):
        records = audit.read_audit_log(str(log_file))

    assert len(records) == 2
    assert records[0]["trace_id"] == "t1"
    assert records[1]["trace_id"] == "t2"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("skipping unparsable audit line" in w.message for w in warnings)


def test_read_audit_log_on_empty_file_returns_empty_list(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    assert audit.read_audit_log(str(empty_file)) == []

