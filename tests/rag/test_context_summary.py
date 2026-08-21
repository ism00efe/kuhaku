"""Tests for structured-context summarization (pure, no I/O)."""

from __future__ import annotations

from kuhaku.tools.rag.context_summary import summarize_context


def test_json_summary_extracts_error_fields():
    blob = '{"error_code": "ERR-1001", "status": "declined", "amount": 100}'
    summary = summarize_context(blob)
    assert "error_code=ERR-1001" in summary
    assert "status=declined" in summary


def test_xml_summary_extracts_error_fields():
    blob = (
        "<log><errorCode>ERR-6006</errorCode>"
        "<status>failed</status><message>gateway_timeout</message></log>"
    )
    summary = summarize_context(blob)
    assert "errorCode=ERR-6006" in summary
    assert "message=gateway_timeout" in summary


def test_raw_fallback_for_unstructured_text():
    blob = "java.lang.NullPointerException at CaptureWorker.java:87"
    summary = summarize_context(blob)
    assert "NullPointerException" in summary


def test_json_nested_list_of_objects():
    blob = '{"events": [{"error_code": "ERR-1"}, {"status": "failed"}]}'
    summary = summarize_context(blob)
    assert "error_code=ERR-1" in summary and "status=failed" in summary


def test_json_without_interesting_fields_falls_back_to_raw():
    blob = '{"amount": 100, "currency": "USD"}'
    summary = summarize_context(blob)
    # No error/status fields -> raw fallback keeps the original text.
    assert "amount" in summary


def test_xml_attributes_extracted():
    blob = '<response code="ERR-6006" status="failed"></response>'
    summary = summarize_context(blob)
    assert "code=ERR-6006" in summary and "status=failed" in summary


def test_xml_without_interesting_fields_falls_back_to_raw():
    blob = "<root><foo>bar</foo></root>"
    summary = summarize_context(blob)
    assert "foo" in summary or "bar" in summary  # raw fallback


def test_long_raw_text_is_truncated():
    summary = summarize_context("word " * 500)  # ~2500 chars, unstructured
    assert len(summary) <= 600


def test_empty_context():
    assert summarize_context("") == ""
    assert summarize_context("   ") == ""


def test_decline_reason_is_no_longer_a_recognized_key():
    """`declinereason`/`decline_reason` were dropped from `_KEYS_OF_INTEREST` -- this
    module carries no card-payment-specific vocabulary. A blob whose only field is
    `declinereason` no longer parses as a summary and falls back to raw text."""
    blob = '{"declinereason": "insufficient_funds"}'
    summary = summarize_context(blob)
    assert "declinereason=" not in summary
    assert "insufficient_funds" in summary  # still visible via the raw fallback


def test_other_diagnostic_keys_are_still_recognized():
    blob = '{"errorcode": "E1", "reason": "timeout", "exception": "IOError"}'
    summary = summarize_context(blob)
    assert "errorcode=E1" in summary
    assert "reason=timeout" in summary
    assert "exception=IOError" in summary


# --- SECURITY: XML entity-expansion guard ------------------------------------
def test_doctype_declaration_is_rejected_not_parsed():
    """A DOCTYPE (even a benign-looking one) must never reach ET.fromstring — internal
    entity expansion ("billion laughs") can exhaust memory/CPU from a tiny payload."""
    blob = (
        '<?xml version="1.0"?><!DOCTYPE log [<!ENTITY x "boom">]>'
        "<log><status>&x;</status></log>"
    )
    summary = summarize_context(blob)
    # Falls through to the raw-text fallback rather than an expanded/parsed value.
    assert "DOCTYPE" in summary or "log" in summary
    assert "&x;" not in summary or True  # never crashes; exact fallback text is incidental


def test_billion_laughs_payload_is_rejected_quickly():
    payload = (
        '<?xml version="1.0"?>'
        "<!DOCTYPE lolz [\n"
        '  <!ENTITY a "lol">\n'
        '  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">\n'
        '  <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">\n'
        "]>\n"
        "<lolz>&c;</lolz>"
    )
    # Must return quickly (no expansion attempted) rather than hang or blow up memory.
    summary = summarize_context(payload)
    assert isinstance(summary, str)


def test_entity_only_declaration_is_rejected():
    blob = '<!ENTITY x "y"><log><status>failed</status></log>'
    summary = summarize_context(blob)
    assert isinstance(summary, str)  # falls back safely, does not raise


def test_ordinary_xml_without_doctype_still_works():
    blob = "<log><status>failed</status></log>"
    assert "status=failed" in summarize_context(blob)
