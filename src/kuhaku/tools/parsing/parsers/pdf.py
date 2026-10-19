"""PDF: text, layout and tables, with OCR for pages that have no text layer.

Two backends:

- ``pdfplumber`` (the ``parsing-pdf`` extra) -- per-line text with positions and font
  sizes, so blocks get bounding boxes, headings are recognized by font size, and tables
  are extracted as rows and cells.
- ``pypdf`` (already a kuhaku dependency) -- text only, one paragraph per blank-line
  separated run. Used when pdfplumber is not installed, with a ``layout_unavailable``
  warning, so a bare install still parses PDFs instead of failing.

A page with no extractable text is treated as scanned. With an OCR engine, the images
embedded in it are recognized; without one, the page is reported as a ``no_text_layer``
warning rather than coming back silently empty. Recognition works on the embedded
images (extracted by pypdf), not on a rendering of the page, so a page whose text was
drawn as vector outlines is not recovered -- that would need a rasterizer.
"""

from __future__ import annotations

import io
import logging
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..document import (
    Block,
    BoundingBox,
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
    page_limit_warning,
)
from ..errors import MissingDependency, ParsingError, is_installed, require
from ..ocr import OCREngine, ocr_unavailable_hint
from ..registry import DocumentSource
from .text import paragraphs

logger = logging.getLogger(__name__)

PDF_EXTRA = "parsing-pdf"
Backend = Literal["auto", "pdfplumber", "pypdf"]

# A line this much larger than the document's body text (median character size) is a
# heading candidate. 1.15 separates a 12pt body from 14pt headings while leaving
# rounding noise in the same font alone.
_HEADING_SIZE_RATIO = 1.15
_HEADING_MAX_CHARS = 150
_BULLET_RE = re.compile(r"^(?:[•●▪◦‣∙·\-–*]|\(?\d{1,3}[.)]|[a-z][.)])\s+")
_ORDERED_RE = re.compile(r"^\(?(?:\d{1,3}|[a-z])[.)]\s+")


@dataclass
class _RawPage:
    number: int
    width: float | None
    height: float | None
    blocks: list[Block] = field(default_factory=list)
    has_text: bool = True


class PdfParser:
    name = "pdf"
    media_types: tuple[str, ...] = ("application/pdf",)
    suffixes: tuple[str, ...] = (".pdf",)

    def __init__(self, *, ocr: OCREngine | None = None, backend: Backend = "auto") -> None:
        """``backend="auto"`` uses pdfplumber when installed and falls back to pypdf
        with a warning; ``"pdfplumber"`` requires it; ``"pypdf"`` never uses it."""

        if backend not in ("auto", "pdfplumber", "pypdf"):
            raise ValueError(f"backend must be 'auto', 'pdfplumber' or 'pypdf', got {backend!r}")
        self.ocr = ocr
        self.backend = backend

    def parse(self, source: DocumentSource) -> ParsedDocument:
        if b"%PDF" not in source.data[:1024]:
            raise ParsingError(f"'{source.name}' is not a PDF (no %PDF signature).")

        warnings: list[ParseWarning] = []
        reader = _LazyReader(source)
        use_plumber = self.backend == "pdfplumber" or (
            self.backend == "auto" and is_installed("pdfplumber")
        )
        try:
            if use_plumber:
                raw_pages, total = _read_with_pdfplumber(source)
            else:
                raw_pages, total = _read_with_pypdf(reader)
        except (ParsingError, MissingDependency):
            raise
        except Exception as exc:
            # Both libraries raise a spread of types (PdfReadError, PSException, plain
            # ValueError/KeyError) for a corrupt file; to a caller they mean one thing.
            raise ParsingError(f"Could not read '{source.name}' as a PDF: {exc}") from exc
        if not use_plumber and self.backend == "auto":
            warnings.append(
                ParseWarning(
                    code="layout_unavailable",
                    message="Read with pypdf: text only -- no tables, headings or bounding "
                    f'boxes. Install pip install "kuhaku[{PDF_EXTRA}]" for layout and tables.',
                )
            )

        if source.max_pages is not None and total > source.max_pages:
            warnings.append(page_limit_warning(total, source.max_pages))
        pages = [self._finish_page(reader, raw, warnings) for raw in raw_pages]
        return ParsedDocument(
            source=source.name,
            media_type="application/pdf",
            pages=tuple(pages),
            warnings=tuple(warnings),
            metadata=_metadata(reader),
        )

    def _finish_page(
        self, reader: _LazyReader, raw: _RawPage, warnings: list[ParseWarning]
    ) -> Page:
        page = Page(number=raw.number, blocks=tuple(raw.blocks), width=raw.width, height=raw.height)
        if raw.has_text:
            return page
        if self.ocr is None:
            warnings.append(
                ParseWarning(
                    code="no_text_layer",
                    message=f"Page {raw.number} has no text layer (scanned?) and was not "
                    f"OCR'd: {ocr_unavailable_hint()}.",
                    page=raw.number,
                )
            )
            return page

        try:
            images = _page_images(reader, raw.number)
        except MissingDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a warning, page kept
            warnings.append(
                ParseWarning(
                    code="ocr_failed",
                    message=f"Could not extract the images on page {raw.number}: {exc}",
                    page=raw.number,
                )
            )
            return page
        if not images:
            warnings.append(
                ParseWarning(
                    code="no_text_layer",
                    message=f"Page {raw.number} has no text layer and no embedded image to OCR.",
                    page=raw.number,
                )
            )
            return page
        provenance = Provenance(reader.source.name, raw.number, ocr=True)
        blocks: list[Block] = []
        for index, image in enumerate(images):
            try:
                text = self.ocr.recognize(image)
            except MissingDependency:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad image must not fail the parse
                warnings.append(
                    ParseWarning(
                        code="ocr_failed",
                        message=f"OCR of image {index + 1} on page {raw.number} failed: {exc}",
                        page=raw.number,
                    )
                )
                continue
            blocks.extend(paragraphs(text, provenance))
        if not blocks:
            warnings.append(
                ParseWarning(
                    code="ocr_empty",
                    message=f"OCR found no text on page {raw.number}.",
                    page=raw.number,
                )
            )
        return replace(page, blocks=tuple(blocks), ocr=True)


# --- pypdf -------------------------------------------------------------------------------


def _pypdf_reader(source: DocumentSource) -> Any:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(source.data))
    if reader.is_encrypted and not reader.decrypt(source.password or ""):
        if source.password is None:
            raise ParsingError(
                f"'{source.name}' is encrypted and needs a password: pass password= to "
                f"DocumentParser.parse."
            )
        raise ParsingError(f"'{source.name}' is encrypted and the password is wrong.")
    return reader


class _LazyReader:
    """One pypdf reader per parse, shared by the text, metadata and OCR-image paths."""

    def __init__(self, source: DocumentSource) -> None:
        self.source = source
        self._reader: Any = None

    def get(self) -> Any:
        if self._reader is None:
            self._reader = _pypdf_reader(self.source)
        return self._reader


def _read_with_pypdf(reader: _LazyReader) -> tuple[list[_RawPage], int]:
    pages: list[_RawPage] = []
    all_pages = reader.get().pages
    limit = reader.source.max_pages
    for index, page in enumerate(all_pages):
        if limit is not None and index >= limit:
            break
        number = index + 1
        text = page.extract_text() or ""
        box = page.mediabox
        pages.append(
            _RawPage(
                number=number,
                width=float(box.width),
                height=float(box.height),
                blocks=paragraphs(text, Provenance(reader.source.name, number)),
                has_text=bool(text.strip()),
            )
        )
    return pages, len(all_pages)


def _page_images(reader: _LazyReader, number: int) -> list[bytes]:
    page = reader.get().pages[number - 1]
    images: list[bytes] = []
    try:
        for image in page.images:
            images.append(image.data)
    except ImportError as exc:
        # pypdf re-encodes most image formats through Pillow.
        raise MissingDependency(
            f'Extracting page images for OCR needs Pillow: pip install "kuhaku[parsing-ocr]" '
            f"({exc})"
        ) from exc
    return images


def _metadata(reader: _LazyReader) -> dict[str, Any]:
    try:
        info = reader.get().metadata
    except Exception:
        logger.debug("could not read PDF metadata", exc_info=True)
        return {}
    metadata: dict[str, Any] = {}
    if info is not None:
        if info.title:
            metadata["title"] = str(info.title)
        if info.author:
            metadata["author"] = str(info.author)
    return metadata


# --- pdfplumber --------------------------------------------------------------------------


@dataclass
class _Line:
    text: str
    bbox: BoundingBox
    size: float


def _read_with_pdfplumber(source: DocumentSource) -> tuple[list[_RawPage], int]:
    pdfplumber = require("pdfplumber", extra=PDF_EXTRA, purpose="PDF layout and table extraction")

    collected: list[tuple[_RawPage, list[_Line], list[Block]]] = []
    open_kwargs = {"password": source.password} if source.password is not None else {}
    with pdfplumber.open(io.BytesIO(source.data), **open_kwargs) as pdf:
        total = len(pdf.pages)
        for index, page in enumerate(pdf.pages):
            if source.max_pages is not None and index >= source.max_pages:
                break
            number = index + 1
            tables = page.find_tables()
            boxes = [table.bbox for table in tables]
            text_page = page.filter(_outside(boxes)) if boxes else page
            lines = [
                _Line(
                    text=line["text"],
                    bbox=BoundingBox(line["x0"], line["top"], line["x1"], line["bottom"]),
                    size=_line_size(line),
                )
                for line in text_page.extract_text_lines(return_chars=True)
                if line["text"].strip()
            ]
            provenance = Provenance(source.name, number)
            others: list[Block] = []
            for table in tables:
                rows = tuple(
                    tuple((cell or "").strip() for cell in row) for row in table.extract()
                )
                if any(any(cell for cell in row) for row in rows):
                    others.append(
                        Table(
                            provenance=replace(provenance, bbox=BoundingBox(*table.bbox)),
                            rows=rows,
                            has_header=True,
                        )
                    )
            has_text = any(c["text"].strip() for c in page.chars)
            if has_text:
                # A scanned page's only image is the scan itself, which OCR replaces.
                for image in page.images:
                    bbox = BoundingBox(image["x0"], image["top"], image["x1"], image["bottom"])
                    others.append(Figure(provenance=replace(provenance, bbox=bbox)))
            raw = _RawPage(
                number=number,
                width=float(page.width),
                height=float(page.height),
                has_text=has_text,
            )
            collected.append((raw, lines, others))

    body_size = _body_size([line for _, lines, _ in collected for line in lines])
    heading_sizes = sorted(
        {
            round(line.size, 1)
            for _, lines, _ in collected
            for line in lines
            if _is_heading(line, body_size)
        },
        reverse=True,
    )
    pages: list[_RawPage] = []
    for raw, lines, others in collected:
        provenance = Provenance(source.name, raw.number)
        blocks = _group_lines(lines, provenance, body_size, heading_sizes) + others
        blocks.sort(key=_reading_order)
        raw.blocks = blocks
        pages.append(raw)
    return pages, total


def _reading_order(block: Block) -> tuple[float, float]:
    bbox = block.provenance.bbox
    return (bbox.y0, bbox.x0) if bbox is not None else (0.0, 0.0)


def _outside(boxes: list[tuple[float, float, float, float]]) -> Callable[[dict[str, Any]], bool]:
    """Keep every object except characters whose centre falls inside a table."""

    def keep(obj: dict[str, Any]) -> bool:
        if obj.get("object_type") != "char":
            return True
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)

    return keep


def _line_size(line: dict[str, Any]) -> float:
    chars = line.get("chars") or []
    return float(statistics.median(c["size"] for c in chars)) if chars else 0.0


def _body_size(lines: list[_Line]) -> float:
    sizes = [line.size for line in lines for _ in range(len(line.text)) if line.size]
    return statistics.median(sizes) if sizes else 0.0


def _is_heading(line: _Line, body_size: float) -> bool:
    return (
        body_size > 0
        and line.size >= body_size * _HEADING_SIZE_RATIO
        and len(line.text) <= _HEADING_MAX_CHARS
        and not _BULLET_RE.match(line.text)
    )


def _union(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        min(b.x0 for b in boxes), min(b.y0 for b in boxes),
        max(b.x1 for b in boxes), max(b.y1 for b in boxes),
    )


def _group_lines(
    lines: list[_Line], provenance: Provenance, body_size: float, heading_sizes: list[float]
) -> list[Block]:
    """Turn positioned lines into headings, lists and paragraphs.

    A new paragraph starts at a vertical gap larger than ~0.6 of a line height; list
    items start at a bullet or number marker, and a following line without a marker
    continues the item above it.
    """

    blocks: list[Block] = []
    para: list[_Line] = []
    items: list[tuple[str, _Line]] = []
    item_lines: list[_Line] = []
    ordered = False

    def flush() -> None:
        nonlocal items, item_lines
        if para:
            blocks.append(
                Paragraph(
                    provenance=replace(provenance, bbox=_union([ln.bbox for ln in para])),
                    content="\n".join(l.text for l in para),
                )
            )
            para.clear()
        if items:
            blocks.append(
                ListBlock(
                    provenance=replace(provenance, bbox=_union([ln.bbox for ln in item_lines])),
                    items=tuple(ListItem(text=text) for text, _ in items),
                    ordered=ordered,
                )
            )
            items, item_lines = [], []

    previous: _Line | None = None
    for line in lines:
        gap_break = previous is not None and (
            line.bbox.y0 - previous.bbox.y1 > 0.6 * max(previous.bbox.y1 - previous.bbox.y0, 1.0)
        )
        if _is_heading(line, body_size):
            flush()
            level = heading_sizes.index(round(line.size, 1)) + 1
            blocks.append(
                Heading(
                    provenance=replace(provenance, bbox=line.bbox),
                    content=line.text.strip(),
                    level=min(level, 6),
                )
            )
            previous = None  # whatever follows a heading starts a new block
            continue
        marker = _BULLET_RE.match(line.text)
        if marker:
            if para:
                flush()
            if not items:
                ordered = bool(_ORDERED_RE.match(line.text))
            items.append((line.text[marker.end():].strip(), line))
            item_lines.append(line)
        elif items and not gap_break:
            text, first = items[-1]
            items[-1] = (f"{text} {line.text.strip()}", first)
            item_lines.append(line)
        else:
            if gap_break or items:
                flush()
            para.append(line)
        previous = line
    flush()
    return blocks
