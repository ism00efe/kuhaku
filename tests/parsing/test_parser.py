"""The facade, the registry, the model's renderings, OCR resolution and observability."""

from __future__ import annotations

import io
import logging
import sys

import pytest

import kuhaku.tools.parsing.ocr as ocr_module
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.observability import bind_trace_id
from kuhaku.tools.parsing import (
    BoundingBox,
    DocumentParser,
    DocumentSource,
    Heading,
    ListBlock,
    ListItem,
    MarkdownParser,
    MissingDependency,
    Page,
    Paragraph,
    ParsedDocument,
    ParserRegistry,
    PdfParser,
    Provenance,
    Table,
    TesseractOCR,
    TextParser,
    UnsupportedFormat,
    default_registry,
    resolve_ocr,
)

from .conftest import FakeOCR, make_pdf, table_page

PROV = Provenance("a.md", 1)


# --- model -------------------------------------------------------------------------------


def test_renderings_preserve_structure() -> None:
    doc = ParsedDocument(
        source="a.md",
        media_type="text/markdown",
        pages=(
            Page(
                number=1,
                blocks=(
                    Heading(provenance=PROV, content="Title", level=2),
                    Paragraph(provenance=PROV, content="Body."),
                    ListBlock(
                        provenance=PROV,
                        items=(ListItem("a"), ListItem("b", level=1), ListItem("c")),
                        ordered=True,
                    ),
                ),
            ),
            Page(
                number=2,
                blocks=(
                    Table(
                        provenance=Provenance("a.md", 2, BoundingBox(0, 0, 1, 1)),
                        rows=(("k", "v"), ("x",)),
                    ),
                ),
            ),
        ),
    )
    assert doc.to_markdown() == (
        "## Title\n\nBody.\n\n1. a\n   1. b\n2. c\n\n| k | v |\n| --- | --- |\n| x |  |"
    )
    assert doc.to_text() == "Title\n\nBody.\n\n- a\n  - b\n- c\n\nk | v\nx | "
    assert doc.to_markdown(page_breaks=True).startswith("<!-- page 1 -->\n\n## Title")
    assert "<!-- page 2 -->\n\n| k | v |" in doc.to_markdown(page_breaks=True)
    assert doc.tables[0].header == ("k", "v") and doc.tables[0].n_columns == 2
    assert doc.title == "Title"


def test_empty_document() -> None:
    doc = DocumentParser(ocr=None).parse(b"", filename="empty.txt")
    assert doc.to_text() == "" and doc.to_markdown() == "" and doc.title is None


# --- registry ----------------------------------------------------------------------------


def test_selection_order_media_type_then_suffix_then_content() -> None:
    registry = default_registry(ocr=None)
    md = DocumentSource(b"# x", "notes.txt", "text/markdown; charset=utf-8")
    assert registry.select(md)[0].name == "markdown"  # explicit type beats the suffix
    assert registry.select(DocumentSource(b"# x", "notes.txt"))[0].name == "text"
    pdf = make_pdf([table_page()])
    assert registry.select(DocumentSource(pdf, "upload.bin"))[1] == "application/pdf"
    assert registry.select(DocumentSource(b"just words", "no_suffix"))[0].name == "text"


def test_unsupported_format() -> None:
    with pytest.raises(UnsupportedFormat, match="No parser for 'blob'"):
        DocumentParser(ocr=None).parse(b"\x00\xff\xfe binary", filename="blob")
    with pytest.raises(UnsupportedFormat):
        DocumentParser(ocr=None).parse(b"PK\x03\x04", filename="archive.zip")


def test_registering_a_parser_replaces_the_earlier_one() -> None:
    class Shouting(TextParser):
        name = "shouting"

        def parse(self, source: DocumentSource) -> ParsedDocument:
            doc = super().parse(source)
            return ParsedDocument(
                source=doc.source, media_type=doc.media_type, pages=doc.pages,
                metadata={"loud": True},
            )

    registry = default_registry(ocr=None)
    registry.register(Shouting())
    doc = DocumentParser(registry=registry).parse(b"hi", filename="a.txt")
    assert doc.metadata == {"loud": True}


def test_registry_and_ocr_together_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="ocr= only configures"):
        DocumentParser(registry=ParserRegistry([MarkdownParser()]), ocr=FakeOCR())


def test_path_input_and_type_errors(tmp_path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide", encoding="utf-8")
    parser = DocumentParser(ocr=None)
    assert parser.parse(path).source == "guide.md"
    assert parser.parse(str(path), filename="renamed.md").source == "renamed.md"
    with pytest.raises(TypeError):
        parser.parse(123)  # type: ignore[arg-type]


# --- OCR resolution ----------------------------------------------------------------------


def test_auto_ocr_needs_both_the_extra_and_the_binary(monkeypatch) -> None:
    monkeypatch.setattr(ocr_module, "is_installed", lambda name: True)
    monkeypatch.setattr(ocr_module.shutil, "which", lambda name: None)
    assert resolve_ocr("auto") is None
    monkeypatch.setattr(ocr_module.shutil, "which", lambda name: "/usr/bin/tesseract")
    assert isinstance(resolve_ocr("auto"), TesseractOCR)
    engine = FakeOCR()
    assert resolve_ocr(engine) is engine
    assert resolve_ocr(None) is None


def test_tesseract_without_the_extra_names_it(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    with pytest.raises(MissingDependency, match=r"kuhaku\[parsing-ocr\]"):
        TesseractOCR().recognize(b"\x89PNG")


def test_tesseract_without_the_binary_says_so(monkeypatch) -> None:
    pytesseract = pytest.importorskip("pytesseract")
    image = pytest.importorskip("PIL.Image")

    def not_found(*_args: object, **_kwargs: object) -> str:
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", not_found)
    buffer = io.BytesIO()
    image.new("L", (4, 4)).save(buffer, format="PNG")
    with pytest.raises(MissingDependency, match="Tesseract binary"):
        TesseractOCR().recognize(buffer.getvalue())


# --- observability -----------------------------------------------------------------------


def test_one_span_and_log_record_per_parse_with_counts(caplog) -> None:
    data = make_pdf([table_page(), table_page()])
    registry = ParserRegistry([PdfParser(backend="pypdf")])
    with caplog.at_level(logging.INFO, logger="kuhaku.tools.parsing"):
        doc = DocumentParser(registry=registry).parse(data, filename="r.pdf", trace_id="t-9")
    assert doc.trace_id == "t-9"
    [record] = [r for r in caplog.records if getattr(r, "step", None) == "parsing"]
    assert record.status == "ok"
    assert (record.parser, record.media_type, record.page_count) == ("pdf", "application/pdf", 2)
    assert record.size_bytes == len(data) and record.warning_count == 0
    assert record.duration_ms >= 0
    # The file name is caller-supplied and stays out of telemetry.
    assert "r.pdf" not in str(record.__dict__)


def test_failures_are_logged_and_reraised(caplog) -> None:
    with (
        caplog.at_level(logging.INFO, logger="kuhaku.tools.parsing"),
        pytest.raises(UnsupportedFormat),
    ):
        DocumentParser(ocr=None).parse(b"\x00\xff", filename="x")
    [record] = [r for r in caplog.records if getattr(r, "step", None) == "parsing"]
    assert record.status == "error"


def test_warnings_are_logged(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="kuhaku.tools.parsing"):
        DocumentParser(ocr=None).parse(b"caf\xe9", filename="a.txt")
    assert any("encoding_detected" in r.getMessage() for r in caplog.records)


def test_outer_trace_id_is_inherited() -> None:
    with bind_trace_id("outer"):
        doc = DocumentParser(ocr=None).parse(b"x", filename="a.txt")
    assert doc.trace_id == "outer"
