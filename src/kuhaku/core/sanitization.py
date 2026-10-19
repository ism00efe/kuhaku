"""Deterministic, rule-based sanitization.

Security requirement (non-negotiable): this runs BEFORE embedding and BEFORE any LLM
call, on both the ingestion path and the user query/log path. Raw sensitive data must
never reach the vector store or a prompt. The LLM is never asked to sanitize.

Design notes
------------
- The text is NFKC-normalized and stripped of zero-width characters first, so a
  fullwidth ``＠`` or an e-mail split by a zero-width space is matched like the plain
  form. The returned text is the normalized text: a masked value cannot be restored
  anyway, and mapping matches back onto the unnormalized original would leave exactly
  the obfuscated forms this step exists to catch.
- Order matters. Credentials embedded in URLs and ``key: value`` secrets go first (an
  URL's ``user:password@host`` otherwise reads as an e-mail), then e-mails, tokens, IPs,
  IBANs, cards, national ids and finally phones, the loosest pattern, so digit-heavy
  values are not misclassified.
- Cards, Turkish IDs (TCKN) and IBANs are checksum-validated (Luhn / TCKN / mod-97) to
  avoid masking arbitrary long numbers. Phones are accepted only in plausible shapes
  (international prefix, separators, or a trunk-``0`` national number), so an
  11-digit order or invoice number is left alone.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------
EMAIL = "[EMAIL]"
TOKEN = "[TOKEN]"
SECRET = "[SECRET]"
IP = "[IP]"
IBAN = "[IBAN]"
CARD = "[CARD]"
NATIONAL_ID = "[NATIONAL_ID]"
PHONE = "[PHONE]"


@dataclass(frozen=True)
class Redaction:
    """How many values of a given kind were masked."""

    label: str
    count: int


# ---------------------------------------------------------------------------
# Normalization shared with the v2 guard's normalizer (security/normalizer.py imports
# these, not the other way round: security/__init__ imports the audit module, which
# imports this one, so a dependency in that direction would be circular).
# ---------------------------------------------------------------------------
ZERO_WIDTH_CHARS = (
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u2060"  # WORD JOINER
    "\u2061"  # FUNCTION APPLICATION
    "\u2062"  # INVISIBLE TIMES
    "\u2063"  # INVISIBLE SEPARATOR
    "\u2064"  # INVISIBLE PLUS
    "\u00ad"  # SOFT HYPHEN
    "\u180e"  # MONGOLIAN VOWEL SEPARATOR
)
ZERO_WIDTH_TABLE = {ord(ch): None for ch in ZERO_WIDTH_CHARS}


def normalize_for_matching(text: str) -> str:
    """NFKC + zero-width removal: the form every pattern below is matched against."""

    return unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH_TABLE)


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


def _iban_ok(candidate: str) -> bool:
    """mod-97 (ISO 7064) IBAN checksum; spaces are ignored.

    Move the first 4 characters to the end, map letters A-Z -> 10-35, and check that the
    resulting integer mod 97 equals 1.
    """

    compact = candidate.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return False
    if not (compact[:2].isalpha() and compact[2:4].isdigit()):
        return False
    rearranged = compact[4:] + compact[:4]
    digits = []
    for ch in rearranged:
        if ch.isdigit():
            digits.append(ch)
        elif "A" <= ch <= "Z":
            digits.append(str(ord(ch) - ord("A") + 10))
        else:
            return False
    return int("".join(digits)) % 97 == 1


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
# scheme://user:password@host -- the password (and user) go, the scheme and host stay.
_URL_CREDENTIALS_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/@:]+:[^\s/@]+@")

_SECRET_KEYS = (
    r"pass(?:word|wd|phrase)?|pwd|secret|client[_-]?secret|api[_-]?key|apikey|"
    r"access[_-]?key(?:[_-]?id)?|secret[_-]?(?:access[_-]?)?key|private[_-]?key|"
    r"auth[_-]?token|access[_-]?token|refresh[_-]?token|token|credentials?|"
    r"şifre|sifre|parola|gizli[_ ]?anahtar"
)
# `password: hunter2`, `"api_key": "abc"`, `SECRET=xyz`. The key and separator stay so
# the text still says what was there. A value that is already a placeholder is left.
_KEY_VALUE_SECRET_RE = re.compile(
    rf"(?i)((?<![\w-])[\"']?(?:[\w-]*[_-])?(?:{_SECRET_KEYS})[\"']?\s*[:=]\s*[\"']?)"
    r"(?!\[[A-Z_]+\])([^\s\"',;}]+)"
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# JWTs, Bearer tokens and common API-key shapes.
_TOKEN_RES = (
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+"),
    # sk-..., sk-ant-api03-..., sk-proj-..., pk_live_... -- the body may contain dashes,
    # and must contain a digit so a hyphenated word starting "sk-" is not a key.
    re.compile(r"\b(?:sk|pk|rk)[-_](?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),  # Groq
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA|AIPA)[A-Z0-9]{16}\b"),  # AWS key id
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])"),  # Google API key
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),  # long hex secrets
)
# An AWS secret access key is 40 base64 characters with no prefix of its own; on its own
# that shape is too common to mask, so it is only taken next to an "aws" mention.
_AWS_SECRET_RE = re.compile(r"(?i)(\baws\b[^\n]{0,40}?[\s:=\"'])([A-Za-z0-9/+]{40})(?![A-Za-z0-9/+])")

_IPV4_RE = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w]|\.\d)"
)
# A dotted quad right after "version"/"v"/"build" is a version number, not an address.
_VERSION_CONTEXT_RE = re.compile(
    r"(?i)(?:\bversion|\bver\.?|\bv|\brelease|\bbuild|\bsürüm|\bsurum|\brev\.?)\s*$"
)
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

# Candidate IBAN: country code, check digits, then the BBAN in groups of four (spaced
# as printed, or run together). Validated by mod-97; an over-long match (a following
# all-caps word) is trimmed group by group before giving up.
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")

# Candidate card: 13-19 digits, optionally grouped by spaces, dashes or dots. Luhn.
# Written digit-first so a trailing separator is never swallowed into the match.
_CARD_RE = re.compile(r"\b\d(?:[ .-]?\d){12,18}\b")
# Candidate TCKN: exactly 11 digits. Validated by the TCKN checksum.
_TCKN_RE = re.compile(r"\b\d{11}\b")
# US SSN, dashed form only (nine bare digits are too common to call an SSN), excluding
# the ranges the SSA never issues.
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")

# Phone candidates: an optional "+", then digits with space/dot/dash/paren separators.
# _phone_ok decides which candidates are plausibly phone numbers.
_PHONE_CANDIDATE_RE = re.compile(
    r"(?<![\w.+/-])(?:\+ ?)?\(?\d[\d ().-]{5,22}\d(?![\w/]|[.-]\d)"
)
_PHONE_GROUP_SPLIT_RE = re.compile(r"[ .()-]+")
_DATE_SHAPE_RE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$|^\d{1,2}[-./]\d{1,2}[-./]\d{2,4}$")
_THOUSANDS_SHAPE_RE = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")
_PLAIN_NATIONAL_PHONE_RE = re.compile(r"^0[1-9]\d{8,9}$")


def _phone_ok(candidate: str) -> bool:
    candidate = candidate.strip()
    digits = re.sub(r"\D", "", candidate)
    if not 7 <= len(digits) <= 15:
        return False
    if candidate.startswith("+"):
        return len(digits) >= 8
    has_separator = bool(re.search(r"[ .()-]", candidate))
    if not has_separator:
        # A bare digit run is a phone only in trunk-prefixed national form (TR 05xx...,
        # UK 07xxx...). Order, invoice and account numbers are bare runs too.
        return bool(_PLAIN_NATIONAL_PHONE_RE.fullmatch(digits))
    if _DATE_SHAPE_RE.match(candidate) or _THOUSANDS_SHAPE_RE.match(candidate):
        return False
    groups = [g for g in _PHONE_GROUP_SPLIT_RE.split(candidate) if g]
    international = digits.startswith("00")
    if international:
        longest = 8
    elif digits.startswith("0"):
        longest = 6  # national forms such as UK "07911 123456"
    else:
        longest = 4
    return all(len(g) <= longest for g in groups)


# ---------------------------------------------------------------------------
# Substitution helpers
# ---------------------------------------------------------------------------
def _sub_count(pattern: re.Pattern[str], repl: str, text: str) -> tuple[str, int]:
    new, n = pattern.subn(repl, text)
    return new, n


def _sub_validated(
    pattern: re.Pattern[str], repl: str, text: str, is_valid: Callable[[str], bool]
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


def _sub_matches(
    pattern: re.Pattern[str], text: str, replace: Callable[[re.Match[str]], str | None]
) -> tuple[str, int]:
    """``replace`` returns the substitute, or ``None`` to keep the match."""

    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        new = replace(match)
        if new is None:
            return match.group(0)
        count += 1
        return new

    return pattern.sub(_replace, text), count


def _valid_iban_prefix(candidate: str) -> str | None:
    """The longest leading part of an :data:`_IBAN_RE` match that passes mod-97.

    Trailing groups are trimmed one at a time: "TR33 ... 26 ABC" overshoots into "ABC".
    """

    while len(candidate.replace(" ", "")) >= 15:
        if _iban_ok(candidate):
            return candidate
        if " " not in candidate:
            return None
        candidate = candidate.rsplit(" ", 1)[0]
    return None


def _replace_iban(match: re.Match[str]) -> str | None:
    valid = _valid_iban_prefix(match.group(0))
    if valid is None:
        return None
    return IBAN + match.group(0)[len(valid) :]


def _replace_ipv4(match: re.Match[str]) -> str | None:
    prefix = match.string[max(0, match.start() - 12) : match.start()]
    if _VERSION_CONTEXT_RE.search(prefix):
        return None
    return IP


def sanitize_text(text: str) -> tuple[str, list[Redaction]]:
    """Mask sensitive values in ``text``.

    Returns the sanitized (NFKC-normalized, zero-width-free) text and a list of
    :class:`Redaction` counts (useful for surfacing "we masked N items" in the UI / logs,
    without revealing the values).
    """

    if not text:
        return text, []

    text = normalize_for_matching(text)
    counts: dict[str, int] = {}

    def bump(label: str, n: int) -> None:
        if n:
            counts[label] = counts.get(label, 0) + n

    text, n = _URL_CREDENTIALS_RE.subn(lambda m: f"{m.group(1)}{SECRET}@", text)
    bump(SECRET, n)
    text, n = _KEY_VALUE_SECRET_RE.subn(lambda m: f"{m.group(1)}{SECRET}", text)
    bump(SECRET, n)
    text, n = _AWS_SECRET_RE.subn(lambda m: f"{m.group(1)}{SECRET}", text)
    bump(SECRET, n)

    text, n = _sub_count(_EMAIL_RE, EMAIL, text)
    bump(EMAIL, n)

    for token_re in _TOKEN_RES:
        text, n = _sub_count(token_re, TOKEN, text)
        bump(TOKEN, n)

    text, n = _sub_matches(_IPV4_RE, text, _replace_ipv4)
    bump(IP, n)
    text, n = _sub_count(_IPV6_RE, IP, text)
    bump(IP, n)

    text, n = _sub_matches(_IBAN_RE, text, _replace_iban)
    bump(IBAN, n)

    text, n = _sub_validated(_CARD_RE, CARD, text, _luhn_ok)
    bump(CARD, n)

    text, n = _sub_validated(_TCKN_RE, NATIONAL_ID, text, _tckn_ok)
    bump(NATIONAL_ID, n)
    text, n = _sub_count(_SSN_RE, NATIONAL_ID, text)
    bump(NATIONAL_ID, n)

    text, n = _sub_matches(
        _PHONE_CANDIDATE_RE, text, lambda m: PHONE if _phone_ok(m.group(0)) else None
    )
    bump(PHONE, n)

    redactions = [Redaction(label, count) for label, count in counts.items()]
    return text, redactions


def sanitize(text: str) -> str:
    """Convenience wrapper returning only the sanitized text."""

    return sanitize_text(text)[0]
