"""Decoding text files whose encoding is not known up front.

Order of evidence, strongest first:

1. a byte-order mark (UTF-8, UTF-16 LE/BE, UTF-32 LE/BE) -- unambiguous;
2. the ``charset`` parameter of the media type the caller passed
   (``text/plain; charset=windows-1254``);
3. for HTML, a ``<meta charset>`` / ``<meta http-equiv="Content-Type">`` declaration in
   the first few kilobytes;
4. strict UTF-8 -- most text today, and a byte sequence that decodes as UTF-8 almost
   never is anything else;
5. a statistical detector (``charset_normalizer``, the ``parsing-text`` extra; it also
   comes with ``requests``), imported only here. UTF-16/32 guesses are refused for
   content without NUL bytes, which short single-byte text (Turkish cp1254) otherwise
   attracts;
6. UTF-8 with undecodable bytes replaced by U+FFFD.

Every path that did not rest on a declaration or valid UTF-8 adds a :class:`ParseWarning`
(``encoding_detected`` or ``decode_errors``), so a guessed decode is never silent.
"""

from __future__ import annotations

import codecs
import importlib
import logging
import re

from .document import ParseWarning

logger = logging.getLogger(__name__)

TEXT_EXTRA = "parsing-text"

_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)
_CHARSET_PARAM = re.compile(r";\s*charset\s*=\s*\"?([\w.:-]+)\"?", re.IGNORECASE)
_META_CHARSET = re.compile(
    rb"<meta[^>]+charset\s*=\s*[\"']?\s*([\w.:-]+)", re.IGNORECASE
)
_HTML_SNIFF_BYTES = 4096
_SINGLE_BYTE = (
    "cp1254",
    "cp1252",
    "cp1250",
    "cp1251",
    "cp1253",
    "cp1257",
    "iso8859_9",
    "iso8859_15",
    "koi8_r",
)
_MULTI_BYTE = ("gb18030", "big5", "shift_jis", "euc_jp", "euc_kr")
_MULTIBYTE_MIN_BYTES = 64


def charset_from_media_type(media_type: str) -> str:
    match = _CHARSET_PARAM.search(media_type or "")
    return match.group(1) if match else ""


def bom_encoding(data: bytes) -> tuple[str, int] | None:
    """``(encoding, BOM length)`` when ``data`` starts with a byte-order mark."""

    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return encoding, len(bom)
    return None


def html_meta_charset(data: bytes) -> str:
    match = _META_CHARSET.search(data[:_HTML_SNIFF_BYTES])
    return match.group(1).decode("ascii", "replace") if match else ""


def decode_text(
    data: bytes, *, charset: str = "", html: bool = False
) -> tuple[str, list[ParseWarning]]:
    """Decode ``data`` (see the module docstring for the order of evidence)."""

    bom = bom_encoding(data)
    if bom is not None:
        encoding, length = bom
        return _decode_declared(data[length:], encoding, "byte-order mark")
    for declared, origin in (
        (charset, "media type"),
        (html_meta_charset(data) if html else "", "<meta charset>"),
    ):
        if declared and _known(declared):
            return _decode_declared(data, declared, origin)
    try:
        return data.decode("utf-8"), []
    except UnicodeDecodeError as exc:
        utf8_error = exc
    detected = detect_encoding(data)
    if detected is not None:
        warning = ParseWarning(
            code="encoding_detected",
            message=f"Content is not valid UTF-8 and declares no encoding; decoded as "
            f"{detected} (detected).",
        )
        return data.decode(detected, errors="replace"), [warning]
    warning = ParseWarning(
        code="decode_errors",
        message=f"Content is not valid UTF-8 ({utf8_error.reason} at byte "
        f"{utf8_error.start}) and its encoding could not be detected; undecodable bytes "
        "were replaced with U+FFFD.",
    )
    return data.decode("utf-8", errors="replace"), [warning]


def detect_encoding(data: bytes) -> str | None:
    """A detector's best guess, or ``None`` (no detector installed, or no guess)."""

    try:
        detector = importlib.import_module("charset_normalizer")
    except ImportError:
        logger.debug("charset_normalizer not installed; install kuhaku[%s]", TEXT_EXTRA)
        return None
    if b"\x00" in data:
        best = detector.from_bytes(data).best()
        return str(best.encoding) if best is not None else None
    # Restricted to the encodings real documents use: unrestricted, a short cp1254 line
    # comes back as cp1006 or big5hkscs. Multi-byte East Asian encodings only compete
    # once there is enough text for the detector's coherence check to mean something.
    candidates = [detector.from_bytes(data, cp_isolation=list(_SINGLE_BYTE)).best()]
    if len(data) >= _MULTIBYTE_MIN_BYTES:
        candidates.append(
            detector.from_bytes(data, cp_isolation=list(_MULTI_BYTE)).best()
        )
    found = [match for match in candidates if match is not None]
    if not found:
        return None
    return str(min(found, key=lambda match: match.chaos).encoding)


def looks_like_text(data: bytes) -> bool:
    """Whether bytes with no recognizable file signature are plausibly text: a BOM, valid
    UTF-8, or no NUL bytes in the first 8 KiB (binary formats are full of them)."""

    if bom_encoding(data) is not None:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return b"\x00" not in data[:8192]
    return True


def _known(encoding: str) -> bool:
    try:
        codecs.lookup(encoding)
    except LookupError:
        return False
    return True


def _decode_declared(
    data: bytes, encoding: str, origin: str
) -> tuple[str, list[ParseWarning]]:
    try:
        return data.decode(encoding), []
    except UnicodeDecodeError as exc:
        warning = ParseWarning(
            code="decode_errors",
            message=f"Content is not valid {encoding} as declared by its {origin} "
            f"({exc.reason} at byte {exc.start}); undecodable bytes were replaced with U+FFFD.",
        )
        return data.decode(encoding, errors="replace"), [warning]
