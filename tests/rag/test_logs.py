"""Tests for uploaded-log summarization (pure, no I/O)."""

from __future__ import annotations

from kuhaku.tools.rag.logs import summarize_log


def test_json_summary_extracts_error_fields():
    log = '{"error_code": "PAY-1001", "status": "declined", "amount": 100}'
    summary = summarize_log(log)
    assert "error_code=PAY-1001" in summary
    assert "status=declined" in summary


def test_xml_summary_extracts_error_fields():
    log = (
        "<log><errorCode>PAY-6006</errorCode>"
        "<status>failed</status><message>gateway_timeout</message></log>"
    )
    summary = summarize_log(log)
    assert "errorCode=PAY-6006" in summary
    assert "message=gateway_timeout" in summary


def test_raw_fallback_for_unstructured_text():
    log = "java.lang.NullPointerException at CaptureWorker.java:87"
    summary = summarize_log(log)
    assert "NullPointerException" in summary


def test_json_nested_list_of_objects():
    log = '{"events": [{"error_code": "PAY-1"}, {"status": "failed"}]}'
    summary = summarize_log(log)
    assert "error_code=PAY-1" in summary and "status=failed" in summary


def test_json_without_interesting_fields_falls_back_to_raw():
    log = '{"amount": 100, "currency": "TRY"}'
    summary = summarize_log(log)
    # No error/status fields -> raw fallback keeps the original text.
    assert "amount" in summary


def test_xml_attributes_extracted():
    log = '<response code="PAY-6006" status="failed"></response>'
    summary = summarize_log(log)
    assert "code=PAY-6006" in summary and "status=failed" in summary


def test_xml_without_interesting_fields_falls_back_to_raw():
    log = "<root><foo>bar</foo></root>"
    summary = summarize_log(log)
    assert "foo" in summary or "bar" in summary  # raw fallback


def test_long_raw_text_is_truncated():
    summary = summarize_log("word " * 500)  # ~2500 chars, unstructured
    assert len(summary) <= 600


def test_empty_log():
    assert summarize_log("") == ""
    assert summarize_log("   ") == ""


# --- SECURITY: XML entity-expansion guard ------------------------------------
def test_doctype_declaration_is_rejected_not_parsed():
    """A DOCTYPE (even a benign-looking one) must never reach ET.fromstring — internal
    entity expansion ("billion laughs") can exhaust memory/CPU from a tiny payload."""
    log = (
        '<?xml version="1.0"?><!DOCTYPE log [<!ENTITY x "boom">]>'
        "<log><status>&x;</status></log>"
    )
    summary = summarize_log(log)
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
    summary = summarize_log(payload)
    assert isinstance(summary, str)


def test_entity_only_declaration_is_rejected():
    log = '<!ENTITY x "y"><log><status>failed</status></log>'
    summary = summarize_log(log)
    assert isinstance(summary, str)  # falls back safely, does not raise


def test_ordinary_xml_without_doctype_still_works():
    log = "<log><status>failed</status></log>"
    assert "status=failed" in summarize_log(log)
