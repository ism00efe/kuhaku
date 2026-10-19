"""The structured document model every parser produces.

A :class:`ParsedDocument` is a list of :class:`Page` objects, each an ordered list of
blocks (:class:`Heading`, :class:`Paragraph`, :class:`ListBlock`, :class:`Table`,
:class:`CodeBlock`, :class:`Figure`). Every block carries a :class:`Provenance` -- which
source, which page, and a bounding box when the format has one -- so a consumer can cite
or highlight where a value came from.

Formats without fixed pages (DOCX, HTML, Markdown, plain text) produce exactly one page.

Problems that did not stop the parse (a scanned page with no OCR engine, a decode error,
a missing optional backend) are collected in :attr:`ParsedDocument.warnings` rather than
raised or silently swallowed: an empty page with no warning would be indistinguishable
from a genuinely empty page.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class BlockKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    FIGURE = "figure"


@dataclass(frozen=True)
class BoundingBox:
    """A rectangle on a page: top-left origin, y grows downward, in the page's native
    units (PDF points for PDFs, pixels for images)."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class Provenance:
    source: str
    page: int
    """1-based page number. Always 1 for formats without fixed pages."""
    bbox: BoundingBox | None = None
    ocr: bool = False
    """The text was recognized from an image, not read from a text layer."""
    region: str | None = None
    """Where on the page, for formats that separate it: ``"header"`` or ``"footer"``
    (DOCX); ``None`` for the main body."""


@dataclass(frozen=True)
class Block:
    """Base of every block. ``text`` is the plain-text rendering; ``to_markdown`` keeps
    the block's structure (heading level, list markers, table grid)."""

    kind: ClassVar[BlockKind]
    provenance: Provenance

    @property
    def text(self) -> str:  # pragma: no cover - every concrete block overrides it
        raise NotImplementedError

    def to_markdown(self) -> str:  # pragma: no cover - every concrete block overrides it
        raise NotImplementedError


@dataclass(frozen=True)
class Heading(Block):
    kind: ClassVar[BlockKind] = BlockKind.HEADING
    content: str = ""
    level: int = 1
    """1 (top) to 6, as in HTML and Markdown."""

    @property
    def text(self) -> str:
        return self.content

    def to_markdown(self) -> str:
        return f"{'#' * max(1, min(self.level, 6))} {_one_line(self.content)}"


@dataclass(frozen=True)
class Paragraph(Block):
    kind: ClassVar[BlockKind] = BlockKind.PARAGRAPH
    content: str = ""

    @property
    def text(self) -> str:
        return self.content

    def to_markdown(self) -> str:
        return self.content


@dataclass(frozen=True)
class ListItem:
    text: str
    level: int = 0
    """Nesting depth, 0 for a top-level item."""


@dataclass(frozen=True)
class ListBlock(Block):
    kind: ClassVar[BlockKind] = BlockKind.LIST
    items: tuple[ListItem, ...] = ()
    ordered: bool = False

    @property
    def text(self) -> str:
        return "\n".join(f"{'  ' * item.level}- {item.text}" for item in self.items)

    def to_markdown(self) -> str:
        lines: list[str] = []
        counters: dict[int, int] = {}
        for item in self.items:
            # Numbering restarts whenever a shallower item closes a nested run.
            counters = {lvl: n for lvl, n in counters.items() if lvl <= item.level}
            counters[item.level] = counters.get(item.level, 0) + 1
            marker = f"{counters[item.level]}." if self.ordered else "-"
            lines.append(f"{'   ' * item.level}{marker} {_one_line(item.text)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Table(Block):
    kind: ClassVar[BlockKind] = BlockKind.TABLE
    rows: tuple[tuple[str, ...], ...] = ()
    """Rows of cell texts. Rows are padded to the widest row by :meth:`grid`."""
    has_header: bool = True
    """The first row is a header row. Markdown has no headerless table, so when this is
    False ``to_markdown`` renders an empty header row above the data."""

    @property
    def n_columns(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    def grid(self) -> list[list[str]]:
        width = self.n_columns
        return [list(row) + [""] * (width - len(row)) for row in self.rows]

    @property
    def header(self) -> tuple[str, ...]:
        return self.rows[0] if self.has_header and self.rows else ()

    @property
    def text(self) -> str:
        return "\n".join(" | ".join(_one_line(c) for c in row) for row in self.grid())

    def to_markdown(self) -> str:
        grid = self.grid()
        if not grid:
            return ""
        width = len(grid[0])
        if self.has_header:
            header, body = grid[0], grid[1:]
        else:
            header, body = [""] * width, grid
        lines = [_md_row(header), "| " + " | ".join(["---"] * width) + " |"]
        lines.extend(_md_row(row) for row in body)
        return "\n".join(lines)


@dataclass(frozen=True)
class CodeBlock(Block):
    kind: ClassVar[BlockKind] = BlockKind.CODE
    content: str = ""
    language: str = ""

    @property
    def text(self) -> str:
        return self.content

    def to_markdown(self) -> str:
        fence = "````" if "```" in self.content else "```"
        return f"{fence}{self.language}\n{self.content}\n{fence}"


@dataclass(frozen=True)
class Figure(Block):
    """An image or figure. Its pixels are not kept -- only what describes it."""

    kind: ClassVar[BlockKind] = BlockKind.FIGURE
    alt: str = ""
    """Alt text or caption, when the source has one."""
    src: str = ""

    @property
    def text(self) -> str:
        return f"[Figure: {self.alt}]" if self.alt else "[Figure]"

    def to_markdown(self) -> str:
        return f"![{_one_line(self.alt)}]({self.src})"


@dataclass(frozen=True)
class ParseWarning:
    code: str
    """Stable identifier, e.g. ``no_text_layer``, ``ocr_unavailable``,
    ``layout_unavailable``, ``decode_errors``, ``ocr_failed``."""
    message: str
    page: int | None = None


@dataclass(frozen=True)
class Page:
    number: int
    blocks: tuple[Block, ...] = ()
    width: float | None = None
    height: float | None = None
    ocr: bool = False
    """The page had no text layer and its text was recognized by OCR."""

    @property
    def text(self) -> str:
        return _join(block.text for block in self.blocks)

    def to_markdown(self) -> str:
        return _join(block.to_markdown() for block in self.blocks)


@dataclass(frozen=True)
class ParsedDocument:
    source: str
    """The file name (or caller-supplied name) the document was parsed from."""
    media_type: str
    """The MIME type the parser was selected for."""
    pages: tuple[Page, ...] = ()
    warnings: tuple[ParseWarning, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None

    @property
    def blocks(self) -> Iterator[Block]:
        for page in self.pages:
            yield from page.blocks

    @property
    def tables(self) -> list[Table]:
        return [block for block in self.blocks if isinstance(block, Table)]

    @property
    def title(self) -> str | None:
        """``metadata["title"]`` when the format carries one, else the first heading."""

        if self.metadata.get("title"):
            return str(self.metadata["title"])
        for block in self.blocks:
            if isinstance(block, Heading):
                return block.content
        return None

    @property
    def ocr_pages(self) -> list[int]:
        return [page.number for page in self.pages if page.ocr]

    def to_text(self) -> str:
        return _join(page.text for page in self.pages)

    def to_markdown(self, *, page_breaks: bool = False) -> str:
        """Render as Markdown -- headings as ``#``, tables as pipe tables -- which is
        the shape ``StructuralChunker`` in the RAG tool splits on.

        ``page_breaks`` puts an HTML comment ``<!-- page N -->`` before each page, so
        page numbers survive into a plain-text consumer without showing up when the
        Markdown is rendered.
        """

        if not page_breaks:
            return _join(page.to_markdown() for page in self.pages)
        return _join(
            f"<!-- page {page.number} -->\n\n{page.to_markdown()}".rstrip()
            for page in self.pages
        )


def page_limit_warning(total: int | None, max_pages: int) -> ParseWarning:
    total_text = f"{total} pages" if total is not None else "more pages"
    return ParseWarning(
        code="page_limit",
        message=f"The document has {total_text}; only the first {max_pages} were read "
        "(max_pages).",
    )


def _join(parts: Iterator[str] | Any) -> str:
    return "\n\n".join(part for part in parts if part.strip())


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(_one_line(c).replace("|", "\\|") for c in cells) + " |"
