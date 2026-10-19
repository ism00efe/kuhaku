"""OCR language resolution, multi-page TIFF, encrypted PDFs, size/page limits, and the
core Document protocol a ParsedDocument satisfies."""

from __future__ import annotations

import io
import sys
import types

import pytest

from kuhaku.core.exceptions import ConfigError
from kuhaku.tools.classification.documents import document_text, is_document
from kuhaku.tools.parsing import (
    DocumentParser,
    DocumentTooLarge,
    ParserRegistry,
    ParsingError,
    TesseractOCR,
    TextParser,
)
from kuhaku.tools.parsing import ocr as ocr_module

from .conftest import FakeOCR, make_pdf, table_page


def _fake_pytesseract(monkeypatch, languages):
    module = types.ModuleType("pytesseract")
    module.get_languages = lambda config="": list(languages)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pytesseract", module)


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        (["eng", "tur", "osd"], "tur+eng"),
        (["eng"], "eng"),
        (["tur"], "tur"),
        (["deu"], "eng"),
    ],
)
def test_auto_ocr_language_prefers_turkish_and_english(
    monkeypatch, installed, expected
):
    _fake_pytesseract(monkeypatch, installed)
    assert TesseractOCR().resolved_lang() == expected
    assert TesseractOCR(lang="deu").resolved_lang() == "deu"


def test_ocr_lang_reaches_the_auto_engine(monkeypatch):
    monkeypatch.setattr(ocr_module, "tesseract_available", lambda: True)
    engine = ocr_module.resolve_ocr("auto", lang="fra")
    assert isinstance(engine, TesseractOCR) and engine.lang == "fra"
    with pytest.raises(ConfigError):
        DocumentParser(registry=ParserRegistry([TextParser()]), ocr_lang="tur")


def _tiff(frames: int) -> bytes:
    image = pytest.importorskip("PIL.Image")
    pages = [image.new("L", (8, 8), color=40 * i) for i in range(frames)]
    buffer = io.BytesIO()
    pages[0].save(buffer, format="TIFF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()


def test_multi_page_tiff_is_one_page_per_frame():
    ocr = FakeOCR("page text")
    doc = DocumentParser(ocr=ocr).parse(_tiff(3), filename="fax.tiff")
    assert [page.number for page in doc.pages] == [1, 2, 3]
    assert len(ocr.calls) == 3 and all(
        call.startswith(b"\x89PNG") for call in ocr.calls
    )
    assert [b.provenance.page for b in doc.blocks] == [1, 2, 3]


def test_max_pages_limits_tiff_frames_and_warns():
    ocr = FakeOCR("x")
    doc = DocumentParser(ocr=ocr, max_pages=2).parse(_tiff(3), filename="fax.tif")
    assert len(doc.pages) == 2 and len(ocr.calls) == 2
    assert [w.code for w in doc.warnings] == ["page_limit"]


def test_max_pages_limits_pdf_pages_and_warns():
    pdf = make_pdf([table_page(), table_page(), table_page()])
    doc = DocumentParser(ocr=None, max_pages=1).parse(pdf, filename="r.pdf")
    assert len(doc.pages) == 1
    assert "page_limit" in [w.code for w in doc.warnings]


def test_max_bytes_refuses_large_content_before_parsing(tmp_path):
    with pytest.raises(DocumentTooLarge):
        DocumentParser(ocr=None, max_bytes=4).parse(b"hello world", filename="a.txt")
    path = tmp_path / "big.txt"
    path.write_bytes(b"x" * 100)
    with pytest.raises(DocumentTooLarge):
        DocumentParser(ocr=None, max_bytes=10).parse(path)
    assert DocumentParser(ocr=None, max_bytes=100).parse(path).to_text() == "x" * 100
    with pytest.raises(ConfigError):
        DocumentParser(ocr=None, max_pages=0)


def _encrypted_pdf(password: str) -> bytes:
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(io.BytesIO(make_pdf([table_page()])))
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_encrypted_pdf_opens_with_its_password():
    try:
        data = _encrypted_pdf("s3cret")
    except Exception as exc:  # noqa: BLE001 - encryption backend missing
        pytest.skip(f"cannot build an encrypted PDF here: {exc}")
    parser = DocumentParser(ocr=None, registry=None)
    with pytest.raises(ParsingError, match="needs a password"):
        parser.parse(data, filename="e.pdf")
    with pytest.raises(ParsingError, match="password is wrong"):
        parser.parse(data, filename="e.pdf", password="nope")
    doc = parser.parse(data, filename="e.pdf", password="s3cret")
    assert doc.to_text().strip()


def test_parsed_document_satisfies_the_core_document_protocol():
    doc = DocumentParser(ocr=None).parse(b"# Title\n\nBody", filename="a.md")
    assert is_document(doc)
    flat = document_text(doc)
    assert flat.source == "a.md" and flat.pages == ((1, "# Title\n\nBody"),)
