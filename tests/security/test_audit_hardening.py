"""audit.record(): one line per record whatever the text, core-side sanitization of
every free-text field, a keyed input hash, and non-JSON metadata."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import date

from kuhaku.core.exceptions import KuhakuError
from kuhaku.core.security import audit


def _record(path, **overrides) -> None:
    kwargs = {"trace_id": "t1", "raw_question": "soru?", "sanitized_retrieval_query": "soru"}
    kwargs.update(overrides)
    audit.record(str(path), **kwargs)


def test_line_separator_in_query_keeps_record_readable(tmp_path):
    log = tmp_path / "audit.jsonl"
    query = "first" + chr(0x2028) + "second" + chr(0x85) + "third"
    _record(log, sanitized_retrieval_query=query, raw_question=query)
    _record(log, trace_id="t2")

    records = audit.read_recent(str(log))
    assert [r["trace_id"] for r in records] == ["t1", "t2"]
    assert records[0]["input_redacted"] == query
    # Escaped on disk: the file has exactly one physical line per record.
    assert len(log.read_bytes().split(b"\n")) == 3


def test_record_sanitizes_every_free_text_field(tmp_path):
    log = tmp_path / "audit.jsonl"
    _record(
        log,
        sanitized_retrieval_query="mail me at ali@example.com",
        event_type="note ali@example.com",
        action="call +90 532 123 45 67",
        target="user ali@example.com",
        metadata={"comment": "password: hunter2", "nested": ["ali@example.com"]},
        output_checks={"detail": "ali@example.com"},
    )
    raw = log.read_text(encoding="utf-8")
    assert "ali@example.com" not in raw
    assert "hunter2" not in raw
    assert "532 123" not in raw
    record = audit.read_recent(str(log))[0]
    assert record["input_redacted"] == "mail me at [EMAIL]"
    assert record["metadata"]["nested"] == ["[EMAIL]"]


def test_digest_valued_keys_are_not_masked(tmp_path):
    log = tmp_path / "audit.jsonl"
    digest = hashlib.sha256(b"prompt").hexdigest()
    _record(
        log,
        metadata={"prompts": {"system": {"name": "x", "hash": digest}}, "sql_sha256": digest},
    )
    record = audit.read_recent(str(log))[0]
    assert record["metadata"]["prompts"]["system"]["hash"] == digest
    assert record["metadata"]["sql_sha256"] == digest


def test_input_hash_is_keyed_not_plain_sha256(tmp_path, monkeypatch):
    monkeypatch.delenv(audit.AUDIT_KEY_ENV, raising=False)
    log = tmp_path / "audit.jsonl"
    _record(log, raw_question="what is my balance?")
    record = audit.read_recent(str(log))[0]
    assert record["input_hash"] != hashlib.sha256(b"what is my balance?").hexdigest()

    key_file = tmp_path / audit.AUDIT_KEY_FILENAME
    assert key_file.exists()
    key = key_file.read_text(encoding="ascii").strip().encode("ascii")
    expected = hmac.new(key, b"what is my balance?", hashlib.sha256).hexdigest()
    assert record["input_hash"] == expected

    # Same directory, same key: two logs of one deployment hash alike.
    other = tmp_path / "other.jsonl"
    _record(other, raw_question="what is my balance?")
    assert audit.read_recent(str(other))[0]["input_hash"] == expected


def test_input_hash_key_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(audit.AUDIT_KEY_ENV, "deployment-secret")
    log = tmp_path / "audit.jsonl"
    _record(log, raw_question="q")
    expected = hmac.new(b"deployment-secret", b"q", hashlib.sha256).hexdigest()
    assert audit.read_recent(str(log))[0]["input_hash"] == expected
    assert not (tmp_path / audit.AUDIT_KEY_FILENAME).exists()


def test_non_json_metadata_is_recorded_as_text(tmp_path):
    log = tmp_path / "audit.jsonl"
    _record(log, metadata={"day": date(2026, 1, 2)})
    record = audit.read_recent(str(log))[0]
    assert record["metadata"]["day"] == "2026-01-02"


def test_audit_write_error_is_a_kuhaku_error(tmp_path):
    assert issubclass(audit.AuditWriteError, KuhakuError)


def test_reader_still_skips_garbage_lines(tmp_path):
    log = tmp_path / "audit.jsonl"
    _record(log)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
    assert len(audit.read_recent(str(log))) == 1
    assert json.loads(log.read_text(encoding="utf-8").split("\n")[0])["trace_id"] == "t1"
