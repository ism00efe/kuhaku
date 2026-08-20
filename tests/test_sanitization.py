"""Tests for the security-critical sanitization module.

These are the guardrail: if masking regresses, PII could reach the vector store or an
LLM. Covers every required category plus false-positive avoidance (checksum validation).
"""

from __future__ import annotations

from kuhaku.core.sanitization import (
    CARD,
    EMAIL,
    IP,
    NATIONAL_ID,
    PHONE,
    TOKEN,
    _luhn_ok,
    _tckn_ok,
    sanitize,
    sanitize_text,
)


def test_email_masked():
    assert sanitize("Contact ahmet.yilmaz@example.com please") == f"Contact {EMAIL} please"


def test_ipv4_masked():
    assert sanitize("client_ip=192.168.14.87 done").count(IP) == 1


def test_ipv6_masked():
    out = sanitize("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334 end")
    assert IP in out
    assert sanitize("loopback is ::1 here") == f"loopback is {IP} here"
    assert sanitize("server at 2001:db8::1 failed") == f"server at {IP} failed"
    assert sanitize("host fe80::1 ping") == f"host {IP} ping"


def test_valid_card_masked():
    # 4111 1111 1111 1111 is a Luhn-valid test PAN.
    out = sanitize("pan 4111 1111 1111 1111 exp")
    assert CARD in out and "4111" not in out


def test_invalid_long_number_not_treated_as_card():
    # 16 digits that fail Luhn must NOT be masked as a card.
    out = sanitize("order 1234567890123456 ref")
    assert CARD not in out


def test_valid_national_id_masked():
    # 10000000146 is a checksum-valid test TCKN (the national ID algorithm this
    # validator implements).
    out = sanitize("national_id 10000000146 ok")
    assert NATIONAL_ID in out and "10000000146" not in out


def test_invalid_national_id_not_masked():
    # 11 digits that fail the TCKN checksum should be left alone.
    out = sanitize("value 12345678901 here")
    assert NATIONAL_ID not in out


def test_turkish_mobile_masked():
    assert PHONE in sanitize("call +90 532 123 45 67 now")
    assert PHONE in sanitize("call 0555 987 65 43 now")


def test_jwt_and_bearer_masked():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk"
    assert TOKEN in sanitize(f"auth {jwt} tail")
    assert TOKEN in sanitize("Authorization: Bearer abc.def-ghi_123")


def test_api_key_masked():
    assert TOKEN in sanitize("key sk-ABCDEFGHIJKLMNOP1234 rest")


def test_report_counts():
    text = "mail a@b.com and a@c.com, ip 10.0.0.1"
    clean, redactions = sanitize_text(text)
    labels = {r.label: r.count for r in redactions}
    assert labels.get(EMAIL) == 2
    assert labels.get(IP) == 1
    assert "a@b.com" not in clean


def test_timestamp_not_masked_as_ip():
    # HH:MM:SS times must not be mistaken for IPv6 addresses.
    text = "event at 2026-07-24T09:15:32Z finished at 10:02:11"
    out = sanitize(text)
    assert IP not in out
    assert "09:15:32" in out and "10:02:11" in out


def test_clean_text_unchanged():
    text = "The payment was declined due to insufficient funds (PAY-1001)."
    assert sanitize(text) == text


def test_empty_input():
    assert sanitize("") == ""
    assert sanitize_text("") == ("", [])


def test_luhn_and_tckn_validators():
    assert _luhn_ok("4111111111111111")
    assert not _luhn_ok("1234567890123456")
    assert _tckn_ok("10000000146")
    assert not _tckn_ok("12345678901")
    assert not _tckn_ok("00000000146")  # first digit cannot be 0
