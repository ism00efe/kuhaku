"""Text decoding: BOM -> media-type charset -> HTML meta charset -> UTF-8 -> detector."""

from __future__ import annotations

import pytest

from kuhaku.tools.parsing import DocumentParser, UnsupportedFormat
from kuhaku.tools.parsing.encoding import decode_text

TURKISH = (
    "Sipariş listesi: 5 kutu çay, 2 kilo şeker. Müşteri: Öztürk Gıda, İstanbul. "
    "Teslimat ılık havada yapılacak; ödeme peşin.\n"
) * 3


def _parse(data: bytes, **kwargs: str):
    return DocumentParser(ocr=None).parse(data, **kwargs)


def test_cp1254_turkish_file_is_detected() -> None:
    doc = _parse(TURKISH.encode("cp1254"), filename="siparis.txt")
    assert "Müşteri: Öztürk Gıda, İstanbul" in doc.to_text()
    assert [w.code for w in doc.warnings] == ["encoding_detected"]


def test_utf16_bom_is_honoured() -> None:
    for encoding in ("utf-16", "utf-16-be", "utf-32"):
        data = TURKISH.encode(encoding)
        if encoding == "utf-16-be":
            data = b"\xfe\xff" + data
        doc = _parse(data, filename="a.txt")
        assert doc.to_text().startswith("Sipariş listesi"), encoding
        assert doc.warnings == ()


def test_media_type_charset_is_honoured() -> None:
    data = "Kısa çay".encode("cp1254")
    doc = _parse(data, filename="a.txt", media_type="text/plain; charset=windows-1254")
    assert doc.to_text() == "Kısa çay" and doc.warnings == ()


def test_html_meta_charset_is_honoured() -> None:
    html = '<html><head><meta charset="windows-1254"></head><body><p>Kısa çay</p></body></html>'
    doc = _parse(html.encode("cp1254"), filename="a.html")
    assert doc.to_text() == "Kısa çay" and doc.warnings == ()
    http_equiv = (
        '<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-9">'
        "<p>ağaç</p>"
    )
    assert _parse(http_equiv.encode("iso-8859-9"), filename="b.htm").to_text() == "ağaç"


def test_non_utf8_bytes_without_a_filename_are_still_text() -> None:
    doc = _parse(TURKISH.encode("cp1254"))
    assert doc.media_type == "text/plain"
    assert "çay" in doc.to_text()
    with pytest.raises(UnsupportedFormat):
        _parse(b"\x00\x01\x02\xff binary")


def test_without_a_detector_bytes_are_replaced_and_reported(monkeypatch) -> None:
    from kuhaku.tools.parsing import encoding

    monkeypatch.setattr(encoding, "detect_encoding", lambda data: None)
    text, warnings = decode_text(b"caf\xe9")
    assert text == "caf�"
    assert [w.code for w in warnings] == ["decode_errors"]


def test_an_unknown_declared_charset_falls_through_to_detection() -> None:
    text, warnings = decode_text(b"ok", charset="no-such-codec")
    assert text == "ok" and warnings == []
