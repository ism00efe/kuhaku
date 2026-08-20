"""Deterministic, rule-based sanitization.

Security requirement (non-negotiable): this runs BEFORE embedding and BEFORE any LLM
call, on both the ingestion path and the user query/log path. Raw sensitive data must
never reach the vector store or a prompt. The LLM is never asked to sanitize.

Design notes
------------
- Order matters. We mask the most specific / structured patterns first (emails, tokens,
  IPs, cards) before looser ones (phones) so digit-heavy values are not misclassified.
- Cards and Turkish IDs (TCKN) are checksum-validated (Luhn / TCKN algorithm) to avoid
  masking arbitrary long numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------
EMAIL = "[EMAIL]"
TOKEN = "[TOKEN]"
IP = "[IP]"
CARD = "[CARD]"
NATIONAL_ID = "[NATIONAL_ID]"
PHONE = "[PHONE]"


@dataclass(frozen=True)
class Redaction:
    """How many values of a given kind were masked."""

    label: str
    count: int


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def _luhn_ok(digits: str) -> bool:
    """Return True if ``digits`` passes the Luhn checksum (credit cards)."""

    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _tckn_ok(digits: str) -> bool:
    """Return True if ``digits`` is a structurally valid Turkish national ID (TCKN)."""

    if len(digits) != 11 or digits[0] == "0":
        return False
    d = [ord(c) - 48 for c in digits]
    tenth = ((sum(d[0:9:2]) * 7) - sum(d[1:8:2])) % 10
    eleventh = sum(d[0:10]) % 10
    return d[9] == tenth and d[10] == eleventh


# ---------------------------------------------------------------------------
# Patterns (applied in order)
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# JWTs, Bearer tokens, and common API-key shapes (sk-..., ghp_..., xoxb-..., long hex).
_TOKEN_RES = (
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"\b(?:sk|pk|rk)[-_][A-Za-z0-9]{16,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),  # long hex secrets
)

_IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
# Require >=4 hextet groups (>=3 colons) for expanded addresses so HH:MM:SS timestamps
# are not matched, or match compressed addresses containing '::'.
_IPV6_RE = re.compile(
    r"(?<![\w:])"
    r"(?:"
    r"(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}"
    r"|"
    r"(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6})?::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6})"
    r"|"
    r"(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6})::"
    r")"
    r"(?![\w:])"
)

# Candidate card: 13-19 digits, optionally grouped by spaces/dashes. Validated by Luhn.
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# Candidate TCKN: exactly 11 digits. Validated by the TCKN checksum.
_TCKN_RE = re.compile(r"\b\d{11}\b")
# Phones: +90..., 0(5xx)..., and generic separated forms. Kept last (loosest).
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?)\d{3}[ .-]?\d{2}[ .-]?\d{2}(?![\w.])"
)


def _sub_count(pattern: re.Pattern[str], repl: str, text: str) -> tuple[str, int]:
    new, n = pattern.subn(repl, text)
    return new, n


def _sub_validated(
    pattern: re.Pattern[str], repl: str, text: str, is_valid
) -> tuple[str, int]:
    """Replace only matches whose digits pass ``is_valid`` (Luhn / TCKN)."""

    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", match.group(0))
        if is_valid(digits):
            count += 1
            return repl
        return match.group(0)

    return pattern.sub(_replace, text), count


def sanitize_text(text: str) -> tuple[str, list[Redaction]]:
    """Mask sensitive values in ``text``.

    Returns the sanitized text and a list of :class:`Redaction` counts (useful for
    surfacing "we masked N items" in the UI / logs, without revealing the values).
    """

    if not text:
        return text, []

    counts: dict[str, int] = {}

    def bump(label: str, n: int) -> None:
        if n:
            counts[label] = counts.get(label, 0) + n

    text, n = _sub_count(_EMAIL_RE, EMAIL, text)
    bump(EMAIL, n)

    for token_re in _TOKEN_RES:
        text, n = _sub_count(token_re, TOKEN, text)
        bump(TOKEN, n)

    text, n = _sub_count(_IPV4_RE, IP, text)
    bump(IP, n)
    text, n = _sub_count(_IPV6_RE, IP, text)
    bump(IP, n)

    text, n = _sub_validated(_CARD_RE, CARD, text, _luhn_ok)
    bump(CARD, n)

    text, n = _sub_validated(_TCKN_RE, NATIONAL_ID, text, _tckn_ok)
    bump(NATIONAL_ID, n)

    text, n = _sub_count(_PHONE_RE, PHONE, text)
    bump(PHONE, n)

    redactions = [Redaction(label, count) for label, count in counts.items()]
    return text, redactions


def sanitize(text: str) -> str:
    """Convenience wrapper returning only the sanitized text."""

    return sanitize_text(text)[0]
