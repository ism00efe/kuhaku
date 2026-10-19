"""DOCX (Word), through ``python-docx`` -- the ``parsing-docx`` extra.

The body is walked as XML, in document order, so nothing python-docx's high-level API
hides is lost: block-level content controls (``w:sdt``) are descended into, tables are
read cell by cell (``w:tc``) with nested tables and content controls inside cells, and a
merged cell -- which python-docx returns once per grid column / row it spans -- is read
once (its other grid positions are empty strings, so rows keep their width).

Headings and lists come from paragraph styles ("Title", "Heading N", "List ...") and
from Word's numbering markup. Whether a list is ordered is resolved from the numbering
definitions (``numId`` -> abstract numbering -> the level's ``numFmt``), for numbering
set on the paragraph or inherited from its style, so a list numbered from the toolbar
("List Paragraph" + ``numPr``) is ordered when its format is a number. Only when no
definition is found is the style name consulted.

Headers and footers are read too, once per distinct header/footer, as blocks whose
provenance has ``region="header"`` / ``"footer"`` -- headers before the body, footers
after it. DOCX has no fixed pages, so the whole document is page 1 and blocks carry no
bounding box.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from ..document import (
    Block,
    Figure,
    Heading,
    ListBlock,
    ListItem,
    Page,
    Paragraph,
    ParsedDocument,
    Provenance,
    Table,
)
from ..errors import ParsingError, require
from ..registry import DocumentSource

DOCX_EXTRA = "parsing-docx"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_HEADING_STYLE_RE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
logger = logging.getLogger(__name__)
_UNORDERED_FORMATS = frozenset({"bullet", "none", ""})


class DocxParser:
    name = "docx"
    media_types: tuple[str, ...] = (_DOCX_MIME,)
    suffixes: tuple[str, ...] = (".docx",)

    def parse(self, source: DocumentSource) -> ParsedDocument:
        docx = require("docx", extra=DOCX_EXTRA, purpose="Parsing DOCX")

        try:
            document = docx.Document(io.BytesIO(source.data))
        except (zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise ParsingError(
                f"'{source.name}' is not a readable DOCX file: {exc}"
            ) from exc

        provenance = Provenance(source.name, 1)
        reader = _Reader(document, _numbering_formats(document))
        headers, footers = _headers_and_footers(document, reader, provenance)
        body = reader.blocks(document.element.body, provenance)

        metadata: dict[str, Any] = {}
        core = document.core_properties
        if core.title:
            metadata["title"] = core.title
        if core.author:
            metadata["author"] = core.author
        return ParsedDocument(
            source=source.name,
            media_type=_DOCX_MIME,
            pages=(Page(number=1, blocks=tuple(headers + body + footers)),),
            metadata=metadata,
        )


class _Reader:
    def __init__(self, document: Any, formats: dict[tuple[str, int], str]) -> None:
        from docx.text.paragraph import Paragraph as DocxParagraph

        self._document = document
        self._formats = formats
        self._paragraph = DocxParagraph

    # --- block level -------------------------------------------------------------------

    def blocks(self, container: Any, provenance: Provenance) -> list[Block]:
        """The blocks of a body/header/footer element, in document order."""

        blocks: list[Block] = []
        pending: list[ListItem] = []
        pending_ordered = False

        def flush_list() -> None:
            nonlocal pending
            if pending:
                blocks.append(
                    ListBlock(
                        provenance=provenance,
                        items=tuple(pending),
                        ordered=pending_ordered,
                    )
                )
                pending = []

        for element in _block_elements(container):
            if element.tag == f"{_W}tbl":
                flush_list()
                rows = self.table_rows(element)
                if rows:
                    blocks.append(
                        Table(provenance=provenance, rows=rows, has_header=True)
                    )
                continue
            paragraph = self._paragraph(element, self._document._body)
            style = _style_name(paragraph)
            text = paragraph.text.strip()
            list_info = self._list_info(paragraph, style)
            if list_info is not None and text:
                level, ordered = list_info
                if pending and ordered != pending_ordered:
                    flush_list()
                pending_ordered = ordered
                pending.append(ListItem(text=text, level=level))
                continue
            flush_list()
            for alt in _image_alts(element):
                blocks.append(Figure(provenance=provenance, alt=alt))
            if not text:
                continue
            heading = _heading_level(style)
            if heading is not None:
                blocks.append(Heading(provenance=provenance, content=text, level=heading))
            else:
                blocks.append(Paragraph(provenance=provenance, content=text))
        flush_list()
        return blocks

    # --- tables ------------------------------------------------------------------------

    def table_rows(self, tbl: Any) -> tuple[tuple[str, ...], ...]:
        from docx.table import Table as DocxTable

        table = DocxTable(tbl, self._document._body)
        # Keyed by id() with the element kept alive as the value, so an id cannot be
        # reused by another proxy element while the table is read.
        seen: dict[int, Any] = {}
        rows: list[tuple[str, ...]] = []
        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                # A merged cell is one w:tc reported at every grid position it spans
                # (gridSpan across, vMerge down); read it once, keep the grid's shape.
                key = id(cell._tc)
                if key in seen:
                    cells.append("")
                    continue
                seen[key] = cell._tc
                cells.append(self.cell_text(cell._tc))
            rows.append(tuple(cells))
        return tuple(rows)

    def cell_text(self, tc: Any) -> str:
        """A cell's paragraphs and nested tables (cells joined by spaces), as one line."""

        parts: list[str] = []
        for element in _block_elements(tc):
            if element.tag == f"{_W}tbl":
                for row in self.table_rows(element):
                    parts.extend(cell for cell in row if cell)
            else:
                text = self._paragraph(element, self._document._body).text.strip()
                if text:
                    parts.append(text)
        return " ".join(" ".join(parts).split())

    # --- lists -------------------------------------------------------------------------

    def _list_info(self, paragraph: Any, style: str) -> tuple[int, bool] | None:
        """``(level, ordered)``, or None when the paragraph is not a list item."""

        num_id, level = _num_pr(paragraph._p.pPr)
        if num_id is None:
            num_id, style_level = _style_num_pr(paragraph.style)
            if level is None:
                level = style_level
        if num_id is not None and num_id != "0":
            fmt = self._formats.get((num_id, level or 0))
            ordered = (
                fmt not in _UNORDERED_FORMATS
                if fmt is not None
                else ("number" in style.lower())
            )
            if level is None:
                level = _style_level(style)
            return level, ordered
        if style.lower().startswith("list") and style.lower() != "list paragraph":
            return _style_level(style), "number" in style.lower()
        return None


def _block_elements(container: Any) -> Iterator[Any]:
    """``w:p`` and ``w:tbl`` children of ``container``, descending into ``w:sdt``
    content controls (block-level ones wrap paragraphs and tables)."""

    for child in container.iterchildren():
        if child.tag in (f"{_W}p", f"{_W}tbl"):
            yield child
        elif child.tag == f"{_W}sdt":
            content = child.find(f"{_W}sdtContent")
            if content is not None:
                yield from _block_elements(content)


def _headers_and_footers(
    document: Any, reader: _Reader, provenance: Provenance
) -> tuple[list[Block], list[Block]]:
    headers: list[Block] = []
    footers: list[Block] = []
    seen: dict[int, Any] = {}
    for section in document.sections:
        for region, parts, out in (
            (
                "header",
                (section.first_page_header, section.header, section.even_page_header),
                headers,
            ),
            (
                "footer",
                (section.first_page_footer, section.footer, section.even_page_footer),
                footers,
            ),
        ):
            for part in parts:
                try:
                    if part.is_linked_to_previous:
                        continue
                    element = part._element
                except (KeyError, AttributeError, NotImplementedError):
                    # A header/footer part the package lacks is simply absent.
                    logger.debug("docx %s part unreadable", region, exc_info=True)
                    continue
                if id(element) in seen:
                    continue
                seen[id(element)] = element
                out.extend(reader.blocks(element, replace(provenance, region=region)))
    return headers, footers


def _numbering_formats(document: Any) -> dict[tuple[str, int], str]:
    """``{(numId, ilvl): numFmt}`` from the numbering part, or {} without one."""

    try:
        numbering = document.part.numbering_part.element
    except (KeyError, NotImplementedError, AttributeError):
        return {}
    abstract: dict[str, dict[int, str]] = {}
    for node in numbering.iterchildren(f"{_W}abstractNum"):
        levels: dict[int, str] = {}
        for lvl in node.iterchildren(f"{_W}lvl"):
            fmt = lvl.find(f"{_W}numFmt")
            levels[int(lvl.get(f"{_W}ilvl", "0"))] = (
                fmt.get(f"{_W}val", "") if fmt is not None else ""
            )
        abstract[node.get(f"{_W}abstractNumId", "")] = levels
    formats: dict[tuple[str, int], str] = {}
    for num in numbering.iterchildren(f"{_W}num"):
        ref = num.find(f"{_W}abstractNumId")
        if ref is None:
            continue
        for level, fmt in abstract.get(ref.get(f"{_W}val", ""), {}).items():
            formats[(num.get(f"{_W}numId", ""), level)] = fmt
        # A level override can change the format for this numId only.
        for override in num.iterchildren(f"{_W}lvlOverride"):
            lvl = override.find(f"{_W}lvl")
            fmt = lvl.find(f"{_W}numFmt") if lvl is not None else None
            if fmt is not None:
                level = int(override.get(f"{_W}ilvl", "0"))
                formats[(num.get(f"{_W}numId", ""), level)] = fmt.get(f"{_W}val", "")
    return formats


def _num_pr(p_pr: Any) -> tuple[str | None, int | None]:
    num_pr = p_pr.numPr if p_pr is not None else None
    if num_pr is None:
        return None, None
    num_id = num_pr.numId.val if num_pr.numId is not None else None
    level = int(num_pr.ilvl.val) if num_pr.ilvl is not None else None
    return (str(num_id) if num_id is not None else None), level


def _style_num_pr(style: Any) -> tuple[str | None, int | None]:
    """Numbering a paragraph inherits from its style (or the styles it is based on)."""

    depth = 0
    while style is not None and depth < 10:
        p_pr = getattr(style.element, "pPr", None)
        num_id, level = _num_pr(p_pr)
        if num_id is not None:
            return num_id, level
        style = style.base_style
        depth += 1
    return None, None


def _style_name(paragraph: Any) -> str:
    return (paragraph.style.name if paragraph.style is not None else "") or ""


def _style_level(style: str) -> int:
    # "List Bullet 2" / "List Number 3" encode the level in the style name.
    digits = re.search(r"(\d)$", style.strip())
    return int(digits.group(1)) - 1 if digits else 0


def _heading_level(style: str) -> int | None:
    if style.lower() == "title":
        return 1
    match = _HEADING_STYLE_RE.match(style.strip())
    return int(match.group(1)) if match else None


def _image_alts(p_element: Any) -> list[str]:
    """Alt text of each inline/anchored picture in the paragraph ("" when it has none)."""

    alts: list[str] = []
    for drawing in p_element.iter(f"{_W}drawing"):
        doc_pr = next(drawing.iter(f"{_WP}docPr"), None)
        alt = ""
        if doc_pr is not None:
            alt = (doc_pr.get("descr") or doc_pr.get("title") or "").strip()
        alts.append(alt)
    alts.extend("" for _ in p_element.iter(f"{_W}pict"))
    return alts
