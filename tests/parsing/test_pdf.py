from __future__ import annotations

import sys

import pytest

from kuhaku.tools.parsing import (
    DocumentParser,
    Heading,
    ListBlock,
    MissingDependency,
    Paragraph,
    ParserRegistry,
    ParsingError,
    PdfParser,
    Table,
)

from .conftest import FailingOCR, FakeOCR, PdfPage, make_pdf, table_page

SCANNED = PdfPage(image=(40, 30))


def _parse(data: bytes, **parser_kwargs: object):
    registry = ParserRegistry([PdfParser(**parser_kwargs)])  # type: ignore[arg-type]
    return DocumentParser(registry=registry).parse(data, filename="report.pdf")


# --- pypdf (always available) -------------------------------------------------------------


def test_pypdf_backend_reads_text_per_page() -> None:
    data = make_pdf([table_page(), PdfPage(texts=[(72, 700, 11, "Second page text")])])
    doc = _parse(data, backend="pypdf")
    assert doc.media_type == "application/pdf"
    assert [p.number for p in doc.pages] == [1, 2]
    assert "Revenue grew" in doc.pages[0].text
    assert doc.pages[1].text == "Second page text"
    assert doc.pages[0].width == 612 and doc.pages[0].height == 792
    assert all(b.provenance.page == p.number for p in doc.pages for b in p.blocks)
    assert doc.warnings == ()  # an explicit backend choice is not something to warn about


def test_auto_without_pdfplumber_falls_back_with_a_warning(monkeypatch) -> None:
    monkeypatch.setattr("kuhaku.tools.parsing.parsers.pdf.is_installed", lambda name: False)
    doc = _parse(make_pdf([table_page()]))
    assert "Revenue grew" in doc.to_text()
    codes = [w.code for w in doc.warnings]
    assert codes == ["layout_unavailable"]
    assert "kuhaku[parsing-pdf]" in doc.warnings[0].message


def test_metadata_title_is_read() -> None:
    doc = _parse(make_pdf([table_page()], title="Q3 Report"), backend="pypdf")
    assert doc.title == "Q3 Report"


def test_scanned_page_without_ocr_is_a_warning_not_silent_empty_text() -> None:
    doc = _parse(make_pdf([table_page(), SCANNED]), backend="pypdf")
    assert doc.pages[1].blocks == ()
    [warning] = [w for w in doc.warnings if w.code == "no_text_layer"]
    assert warning.page == 2
    assert "kuhaku[parsing-ocr]" in warning.message


def test_scanned_page_is_routed_to_ocr(fake_ocr: FakeOCR) -> None:
    pytest.importorskip("PIL")  # pypdf re-encodes the embedded image through Pillow
    doc = _parse(make_pdf([table_page(), SCANNED]), backend="pypdf", ocr=fake_ocr)
    assert len(fake_ocr.calls) == 1  # only the page without a text layer
    assert fake_ocr.calls[0][:4] == b"\x89PNG"
    page = doc.pages[1]
    assert page.ocr and doc.ocr_pages == [2]
    assert [b.text for b in page.blocks] == ["Recognized line one.", "Recognized line two."]
    assert all(b.provenance.ocr and b.provenance.page == 2 for b in page.blocks)
    assert not doc.pages[0].ocr
    assert doc.warnings == ()


def test_ocr_failure_on_one_page_is_reported() -> None:
    pytest.importorskip("PIL")
    doc = _parse(make_pdf([SCANNED]), backend="pypdf", ocr=FailingOCR())
    assert [w.code for w in doc.warnings] == ["ocr_failed", "ocr_empty"]
    assert "engine crashed" in doc.warnings[0].message


def test_page_without_text_or_image_is_reported_even_with_ocr(fake_ocr: FakeOCR) -> None:
    doc = _parse(make_pdf([PdfPage()]), backend="pypdf", ocr=fake_ocr)
    assert fake_ocr.calls == []
    assert [w.code for w in doc.warnings] == ["no_text_layer"]
    assert "no embedded image" in doc.warnings[0].message


def test_not_a_pdf_raises_parsing_error() -> None:
    with pytest.raises(ParsingError, match="no %PDF signature"):
        _parse(b"hello", backend="pypdf")


def test_corrupt_pdf_raises_parsing_error() -> None:
    with pytest.raises(ParsingError, match="Could not read"):
        _parse(b"%PDF-1.4\n garbage without objects", backend="pypdf")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        PdfParser(backend="pymupdf")  # type: ignore[arg-type]


# --- pdfplumber (parsing-pdf extra) --------------------------------------------------------


def test_pdfplumber_recovers_headings_paragraphs_lists_and_tables() -> None:
    pytest.importorskip("pdfplumber")
    doc = _parse(make_pdf([table_page()]), backend="pdfplumber")
    blocks = doc.pages[0].blocks
    assert [type(b) for b in blocks] == [Heading, Paragraph, Paragraph, ListBlock, Table]
    heading, first, second, bullets, table = blocks
    assert heading.text == "Quarterly Report" and heading.level == 1
    assert first.text.startswith("Revenue grew") and "Costs stayed flat" in first.text
    assert second.text.startswith("A second paragraph")
    assert [i.text for i in bullets.items] == ["first point", "second point"]
    assert table.rows == (("Region", "Sales"), ("North", "120"), ("South", "95"))
    # Table cells are not duplicated as loose text.
    assert "North" not in first.text + second.text
    for block in blocks:
        assert block.provenance.bbox is not None
    assert table.provenance.bbox.x0 == pytest.approx(72) and table.provenance.bbox.x1 == pytest.approx(472)
    assert "| Region | Sales |" in doc.to_markdown()
    assert doc.warnings == ()


def test_pdfplumber_routes_scanned_pages_to_ocr(fake_ocr: FakeOCR) -> None:
    pytest.importorskip("pdfplumber")
    pytest.importorskip("PIL")
    doc = _parse(make_pdf([table_page(), SCANNED]), backend="pdfplumber", ocr=fake_ocr)
    assert doc.ocr_pages == [2]
    assert doc.pages[1].text == "Recognized line one.\n\nRecognized line two."


def test_explicit_pdfplumber_backend_names_the_extra_when_missing(monkeypatch) -> None:
    # A None entry in sys.modules makes the import raise ImportError, installed or not.
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    with pytest.raises(MissingDependency, match=r"kuhaku\[parsing-pdf\]"):
        _parse(make_pdf([table_page()]), backend="pdfplumber")
