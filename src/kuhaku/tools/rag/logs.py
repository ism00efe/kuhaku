"""Lightweight parsing of uploaded JSON/XML payment logs.

Extracts a short, human-readable summary (error codes, status, messages) that augments
the retrieval query. Deliberately forgiving: anything it can't parse is passed through as
raw text. Sanitization happens separately in the query path — this module never needs to
see clean data to work, but it must not assume it either.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

# Field names that commonly carry troubleshooting signal in payment logs.
_KEYS_OF_INTEREST = {
    "errorcode", "error_code", "code", "status", "statuscode", "status_code",
    "responsecode", "response_code", "message", "error", "reason", "detail",
    "exception", "declinereason", "decline_reason", "result", "resultcode",
}


def _walk_json(obj: object, found: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                _walk_json(value, found)
            elif key.lower().replace(" ", "") in _KEYS_OF_INTEREST:
                found.setdefault(key, str(value))
    elif isinstance(obj, list):
        for item in obj:
            _walk_json(item, found)


def _summarize_json(text: str) -> str | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    found: dict[str, str] = {}
    _walk_json(data, found)
    if not found:
        return None
    return "; ".join(f"{k}={v}" for k, v in found.items())


def _summarize_xml(text: str) -> str | None:
    # SECURITY: reject any DOCTYPE/ENTITY declaration before parsing. Python's
    # ElementTree does not resolve *external* entities (unlike lxml), so this isn't
    # classic XXE file-read/SSRF — but internal entity expansion ("billion laughs") can
    # still blow up memory/CPU from a tiny payload. Rejecting outright is simpler and
    # safer than trying to bound expansion, and the caller falls back to treating the
    # text as an opaque raw string, which is still useful.
    if re.search(r"<!(DOCTYPE|ENTITY)\b", text, re.IGNORECASE):
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    found: dict[str, str] = {}
    for elem in root.iter():
        tag = re.sub(r"\{.*\}", "", elem.tag)  # strip namespace, keep original case
        if tag.lower() in _KEYS_OF_INTEREST and elem.text and elem.text.strip():
            found.setdefault(tag, elem.text.strip())
        for attr, value in elem.attrib.items():
            if attr.lower() in _KEYS_OF_INTEREST:
                found.setdefault(attr, value)
    if not found:
        return None
    return "; ".join(f"{k}={v}" for k, v in found.items())


def summarize_log(text: str) -> str:
    """Return a compact summary of a log, or a trimmed raw fallback.

    Note: the returned string may still contain sensitive values — callers must sanitize
    it before embedding or sending to an LLM.
    """

    text = (text or "").strip()
    if not text:
        return ""
    return (
        _summarize_json(text)
        or _summarize_xml(text)
        or " ".join(text.split())[:600]  # raw fallback (e.g. stack traces)
    )
