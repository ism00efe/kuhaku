from __future__ import annotations

import io
import sys

import pytest

from kuhaku.tools.parsing import (
    DocumentParser,
    Figure,
    Heading,
    ListBlock,
    MissingDependency,
    Paragraph,
    ParsingError,
    Table,
)

from .conftest import FakeOCR

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _png() -> bytes:
    image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image.new("L", (8, 8), color=128).save(buffer, format="PNG")
    return buffer.getvalue()


def _docx() -> bytes:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.core_properties.title = "Travel Policy"
    document.add_heading("Travel Policy", level=0)
    document.add_heading("Booking", level=2)
    document.add_paragraph("Book through the portal.")
    document.add_paragraph("Economy class", style="List Bullet")
    document.add_paragraph("Window seat", style="List Bullet 2")
    document.add_paragraph("Ask", style="List Number")
    document.add_paragraph("Book", style="List Number")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Class"
    table.rows[0].cells[1].text = "Limit"
    table.rows[1].cells[0].text = "Economy"
    table.rows[1].cells[1].text = "Any"
    document.add_picture(io.BytesIO(_png()))
    document.add_paragraph("Closing note.")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_docx_blocks_in_document_order() -> None:
    doc = DocumentParser(ocr=None).parse(_docx(), filename="travel.docx")
    assert doc.media_type == DOCX_MIME
    assert doc.title == "Travel Policy"
    blocks = list(doc.blocks)
    assert [type(b) for b in blocks] == [
        Heading, Heading, Paragraph, ListBlock, ListBlock, Table, Figure, Paragraph,
    ]
    assert (blocks[0].level, blocks[1].text, blocks[1].level) == (1, "Booking", 2)
    bullets, numbered = blocks[3], blocks[4]
    assert not bullets.ordered
    assert [(i.text, i.level) for i in bullets.items] == [("Economy class", 0), ("Window seat", 1)]
    assert numbered.ordered and [i.text for i in numbered.items] == ["Ask", "Book"]
    assert blocks[5].rows == (("Class", "Limit"), ("Economy", "Any"))
    assert len(doc.pages) == 1 and all(b.provenance.bbox is None for b in blocks)


def test_docx_is_selected_by_content_without_a_suffix() -> None:
    doc = DocumentParser(ocr=None).parse(_docx(), filename="upload")
    assert doc.media_type == DOCX_MIME


def test_docx_missing_extra_names_it(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "docx", None)
    with pytest.raises(MissingDependency, match=r"kuhaku\[parsing-docx\]"):
        DocumentParser(ocr=None).parse(b"PK\x03\x04", filename="a.docx")


def test_corrupt_docx_raises_parsing_error() -> None:
    pytest.importorskip("docx")
    with pytest.raises(ParsingError):
        DocumentParser(ocr=None).parse(b"not a zip at all", filename="a.docx")


def test_image_with_ocr(fake_ocr: FakeOCR) -> None:
    png = _png()
    doc = DocumentParser(ocr=fake_ocr).parse(png, filename="scan.png")
    assert fake_ocr.calls == [png]
    assert doc.media_type == "image/png"
    assert doc.ocr_pages == [1]
    assert doc.to_text() == "Recognized line one.\n\nRecognized line two."
    assert doc.metadata["ocr_engine"] == "fake-ocr"


def test_image_without_ocr_is_a_placeholder_and_a_warning() -> None:
    doc = DocumentParser(ocr=None).parse(_png(), filename="scan.png")
    assert [type(b) for b in doc.blocks] == [Figure]
    assert [w.code for w in doc.warnings] == ["ocr_unavailable"]
    assert "kuhaku[parsing-ocr]" in doc.warnings[0].message


def test_image_with_no_recognized_text_is_reported() -> None:
    doc = DocumentParser(ocr=FakeOCR(text="  \n")).parse(_png(), filename="blank.png")
    assert list(doc.blocks) == []
    assert [w.code for w in doc.warnings] == ["ocr_empty"]
