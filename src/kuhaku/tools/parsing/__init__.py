"""Document parsing: PDF, DOCX, HTML, Markdown, text and images -> one structured model.

Built only on ``kuhaku.core`` (observability, errors); it imports nothing from the other
tools, and nothing here needs an LLM. Heavy backends are optional extras, imported only
when a file of that type is parsed:

- ``kuhaku[parsing-pdf]`` -- pdfplumber: PDF layout, headings and tables (without it,
  PDFs are still read as text through pypdf, with a ``layout_unavailable`` warning)
- ``kuhaku[parsing-docx]`` -- python-docx: DOCX
- ``kuhaku[parsing-ocr]`` -- pytesseract + Pillow: images and scanned PDF pages
  (also needs the ``tesseract`` binary)

HTML, Markdown and plain text need nothing extra.

    from kuhaku.tools.parsing import DocumentParser

    doc = DocumentParser().parse("report.pdf")
    doc.to_markdown()      # headings as #, tables as pipe tables
    doc.tables             # every Table block, as rows of cells
    doc.warnings           # e.g. a scanned page that could not be OCR'd
"""

from .document import (
    Block,
    BlockKind,
    BoundingBox,
    CodeBlock,
    Figure,
    Heading,
    ListBlock,
    ListItem,
    Page,
    Paragraph,
    ParsedDocument,
    ParseWarning,
    Provenance,
    Table,
)
from .errors import DocumentTooLarge, MissingDependency, ParsingError, UnsupportedFormat
from .ocr import OCREngine, TesseractOCR, resolve_ocr
from .parser import DocumentParser, default_registry
from .parsers import (
    DocxParser,
    HTMLParser,
    ImageParser,
    MarkdownParser,
    PdfParser,
    TextParser,
)
from .registry import DocumentSource, Parser, ParserRegistry

__all__ = [
    "Block",
    "BlockKind",
    "BoundingBox",
    "CodeBlock",
    "DocumentParser",
    "DocumentSource",
    "DocumentTooLarge",
    "DocxParser",
    "Figure",
    "HTMLParser",
    "Heading",
    "ImageParser",
    "ListBlock",
    "ListItem",
    "MarkdownParser",
    "MissingDependency",
    "OCREngine",
    "Page",
    "Paragraph",
    "ParseWarning",
    "ParsedDocument",
    "Parser",
    "ParserRegistry",
    "ParsingError",
    "PdfParser",
    "Provenance",
    "Table",
    "TesseractOCR",
    "TextParser",
    "UnsupportedFormat",
    "default_registry",
    "resolve_ocr",
]
