"""Sanitization coverage: secret/PII shapes the first rule set missed, obfuscated forms,
and values it over-masked."""

from __future__ import annotations

import pytest

from kuhaku.core.sanitization import (
    CARD,
    EMAIL,
    IBAN,
    IP,
    NATIONAL_ID,
    PHONE,
    SECRET,
    TOKEN,
    sanitize,
    sanitize_text,
)
from kuhaku.core.security.output_guard import check_pii_egress


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-AbCdEf1234567890_xyzXYZ-abc",
        "sk-proj-AbCdEf1234567890xyz-QWERTY12",
        "gsk_ABCDEFGHIJKLMNOPQRSTUVWX12",
        "AKIAIOSFODNN7EXAMPLE",
        "AIzaSyA1234567890abcdefghijklmnopqrstuv",
        "github_pat_11ABCDEFG0123456789_abcdefghijkl",
    ],
)
def test_provider_api_keys_masked_whole(secret):
    assert sanitize(f"key {secret} end") == f"key {TOKEN} end"


def test_aws_secret_access_key_masked():
    out = sanitize("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    assert "wJalr" not in out and SECRET in out


@pytest.mark.parametrize(
    ("text", "tail"),
    [
        ("pay to TR33 0006 1005 1978 6457 8413 26 thanks", " thanks"),
        ("pay to TR330006100519786457841326 thanks", " thanks"),
        ("pay to GB82 WEST 1234 5698 7654 32 OK", " OK"),
    ],
)
def test_iban_masked_plain_and_spaced(text, tail):
    assert sanitize(text) == f"pay to {IBAN}{tail}"


def test_invalid_iban_not_masked():
    assert IBAN not in sanitize("ref TR00 0000 0000 0000 0000 0000 00")


def test_output_guard_flags_spaced_iban():
    assert check_pii_egress("IBAN: TR33 0006 1005 1978 6457 8413 26", []) == ["IBAN"]


def test_us_ssn_masked():
    assert sanitize("ssn 123-45-6789 here") == f"ssn {NATIONAL_ID} here"
    assert NATIONAL_ID not in sanitize("ssn 000-12-3456 here")


@pytest.mark.parametrize(
    "phone",
    [
        "+44 20 7946 0958",
        "020 7946 0958",
        "07911 123456",
        "+33 1 23 45 67 89",
        "06 12 34 56 78",
        "05321234567",
        "(212) 555-0100",
    ],
)
def test_uk_eu_phone_forms_masked(phone):
    assert sanitize(f"call {phone} now") == f"call {PHONE} now"


def test_email_obfuscated_by_zero_width_or_fullwidth_masked():
    assert sanitize("mail ali\u200b@exam\u200dple.com ok") == f"mail {EMAIL} ok"
    assert sanitize("mail ａｌｉ＠ｅｘａｍｐｌｅ．ｃｏｍ ok") == f"mail {EMAIL} ok"


def test_card_with_dot_separators_masked_without_eating_the_space():
    assert sanitize("card 4111.1111.1111.1111 x") == f"card {CARD} x"
    assert sanitize("pan 4111 1111 1111 1111 exp") == f"pan {CARD} exp"


def test_url_credentials_masked_host_kept():
    out = sanitize("db postgres://admin:hunter2@db.example.com:5432/app")
    assert out == f"db postgres://{SECRET}@db.example.com:5432/app"


def test_key_value_secrets_masked():
    out, redactions = sanitize_text(
        'password: hunter2 and "api_key": "abc123" and DB_PASSWORD=s3cr3t'
    )
    assert "hunter2" not in out and "abc123" not in out and "s3cr3t" not in out
    assert out.startswith(f"password: {SECRET}")
    assert {r.label: r.count for r in redactions} == {SECRET: 3}


def test_version_numbers_not_masked_as_ip():
    out = sanitize("version 1.2.3.4 and v1.2.3.4 but ip 10.0.0.1")
    assert out == f"version 1.2.3.4 and v1.2.3.4 but ip {IP}"


def test_order_and_invoice_numbers_not_masked_as_phone():
    text = "order 20240512345 invoice 12345678901 date 2024-05-12 total 1.234.567 TL"
    assert sanitize(text) == text


def test_ordinary_config_text_untouched():
    text = "max_tokens: 512, temperature 0.7, pages 120-135, at 12:30:45"
    assert sanitize(text) == text
