"""HTML, through the standard library's ``html.parser``.

The standard library tokenizer is lenient enough for real-world markup (unclosed
``<p>``/``<li>``, stray end tags) and costs no dependency, which is why this parser does
not use BeautifulSoup. It walks the token stream once and emits a block whenever a
block-level element closes; inline elements only contribute text.

The tokenizer does not build a tree, so the HTML5 rules for elements whose end tag may
be omitted are applied here: a ``<td>``/``<th>`` closes the open cell, a ``<tr>`` (or a
``<thead>``/``<tbody>``/``<tfoot>``) closes the open cell and row, and ``</table>``
closes all of them. Without that, ``<tr><td>1<td>2</table>`` -- valid HTML -- would
leave a cell open and swallow the rest of the document. A table still open at the end
of the input is emitted and reported as an ``unclosed_table`` warning.
"""

from __future__ import annotations

from html.parser import HTMLParser as _StdlibHTMLParser

from ..document import (
    Block,
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
from ..registry import DocumentSource
from .text import decode_text

# Never rendered as content.
_SKIP = {"script", "style", "noscript", "template", "head", "svg", "math", "iframe", "object"}
# Foreign content whose <title> is an element's tooltip, not the document's title.
_FOREIGN = {"svg", "math"}
# Close the paragraph being collected (but are not themselves a block).
_BREAKS = {
    "p", "div", "section", "article", "main", "header", "footer", "aside", "nav",
    "blockquote", "figure", "figcaption", "form", "fieldset", "address", "dl", "dt",
    "dd", "hr", "body", "html", "details", "summary",
}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_VOID = {"br", "img", "hr", "meta", "link", "input", "area", "base", "col", "embed",
         "source", "track", "wbr"}
_SECTIONS = {"thead", "tbody", "tfoot"}


class _TableState:
    def __init__(self) -> None:
        self.rows: list[list[str]] = []
        self.row_open = False
        self.row_is_header = False
        self.has_header = False
        self.cell: list[str] | None = None
        self.caption: list[str] | None = None
        self.captions: list[str] = []


class _Walker(_StdlibHTMLParser):
    def __init__(self, provenance: Provenance) -> None:
        super().__init__(convert_charrefs=True)
        self.provenance = provenance
        self.blocks: list[Block] = []
        self.warnings: list[ParseWarning] = []
        self.title = ""
        self._title_seen = False
        self._skip_depth = 0
        self._foreign_depth = 0
        self._in_title = False
        self._text: list[str] = []
        self._heading: int | None = None
        self._pre_depth = 0
        self._pre_lang = ""
        # One entry per open <ul>/<ol> (True for <ol>); its length is the nesting depth.
        # Nested lists flatten into the outermost list's items, each with its level.
        self._lists: list[bool] = []
        self._list_out: list[ListItem] = []
        self._list_ordered = False
        # One entry per open <table>. A table nested inside a cell flattens into that
        # cell's text: ``_nested`` counts how deep inside the cell we are.
        self._tables: list[_TableState] = []
        self._nested = 0

    # --- emission ----------------------------------------------------------------------

    def _emit(self, block: Block) -> None:
        # A block inside a list item (a table in an <li>) must come after the items
        # before it, not ahead of the whole list: emit the list so far, then the block;
        # later items start a new list block.
        if self._list_out:
            self._emit_list()
        self.blocks.append(block)

    def _collected(self) -> str:
        raw = "".join(self._text)
        self._text.clear()
        return raw if self._pre_depth else " ".join(raw.split())

    def _flush_paragraph(self) -> None:
        text = self._collected()
        if not text:
            return
        if self._heading is not None:
            # An unclosed heading (<h2>Title<p>...) ends where the next block starts.
            level, self._heading = self._heading, None
            self._emit(Heading(provenance=self.provenance, content=text, level=level))
        elif self._lists:
            self._list_out.append(ListItem(text=text, level=len(self._lists) - 1))
        else:
            self.blocks.append(Paragraph(provenance=self.provenance, content=text))

    def _emit_list(self) -> None:
        if self._list_out:
            self.blocks.append(
                ListBlock(
                    provenance=self.provenance,
                    items=tuple(self._list_out),
                    ordered=self._list_ordered,
                )
            )
        self._list_out = []

    # --- tables ------------------------------------------------------------------------

    @property
    def _table(self) -> _TableState | None:
        return self._tables[-1] if self._tables else None

    def _close_cell(self) -> None:
        table = self._table
        if table is not None and table.cell is not None:
            table.rows[-1].append(" ".join("".join(table.cell).split()))
            table.cell = None

    def _end_row(self) -> None:
        table = self._table
        if table is None:
            return
        self._close_cell()
        if table.row_open:
            if len(table.rows) == 1 and table.row_is_header and table.rows[0]:
                table.has_header = True
            table.row_open = False

    def _start_row(self) -> None:
        table = self._table
        assert table is not None
        self._end_row()
        table.rows.append([])
        table.row_open = True
        table.row_is_header = True

    def _start_cell(self, tag: str) -> None:
        table = self._table
        assert table is not None
        if not table.row_open:
            self._start_row()  # a cell directly under <table>/<tbody> opens a row
        self._close_cell()
        table.cell = []
        if tag == "td":
            table.row_is_header = False

    def _end_table(self) -> None:
        self._end_row()
        table = self._tables.pop()
        for caption in table.captions:
            self._emit(Paragraph(provenance=self.provenance, content=caption))
        rows = [tuple(row) for row in table.rows if row]
        if rows:
            self._emit(Table(provenance=self.provenance, rows=tuple(rows), has_header=table.has_header))

    # --- tokens ------------------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title" and not self._foreign_depth:
            self._in_title = not self._title_seen
            return
        if tag in _SKIP:
            if tag not in _VOID:
                self._skip_depth += 1
                if tag in _FOREIGN:
                    self._foreign_depth += 1
            return
        if self._skip_depth:
            return
        table = self._table
        if table is not None and table.caption is not None:
            if tag == "br":
                table.caption.append(" ")
            return
        if table is not None and table.cell is not None:
            self._start_in_cell(tag, attrs)
            return
        if tag in _HEADINGS:
            self._flush_paragraph()
            self._heading = _HEADINGS[tag]
        elif tag == "pre":
            self._flush_paragraph()
            self._pre_depth += 1
            self._pre_lang = ""
        elif tag == "code" and self._pre_depth:
            for name, value in attrs:
                if name == "class" and value:
                    for cls in value.split():
                        if cls.startswith("language-"):
                            self._pre_lang = cls.removeprefix("language-")
        elif tag in ("ul", "ol"):
            # Inside a list this flushes the enclosing item's text before its sublist.
            self._flush_paragraph()
            if not self._lists:
                self._list_ordered = tag == "ol"
            self._lists.append(tag == "ol")
        elif tag == "li":
            self._flush_paragraph()
        elif tag == "table":
            self._flush_paragraph()
            self._tables.append(_TableState())
        elif tag == "caption" and table is not None:
            table.caption = []
        elif tag == "tr" and table is not None:
            self._start_row()
        elif tag in _SECTIONS and table is not None:
            self._end_row()
        elif tag in ("td", "th") and table is not None:
            self._start_cell(tag)
        elif tag == "img":
            self._flush_paragraph()
            values = dict(attrs)
            self._emit(
                Figure(
                    provenance=self.provenance,
                    alt=(values.get("alt") or "").strip(),
                    src=values.get("src") or "",
                )
            )
        elif tag == "br":
            self._text.append("\n" if self._pre_depth else " ")
        elif tag in _BREAKS:
            self._flush_paragraph()

    def _start_in_cell(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        table = self._table
        assert table is not None and table.cell is not None
        if not self._nested:
            if tag in ("td", "th"):
                self._start_cell(tag)
                return
            if tag == "tr":
                self._start_row()
                return
            if tag in _SECTIONS:
                self._end_row()
                return
        if tag == "img":
            alt = (dict(attrs).get("alt") or "").strip()
            table.cell.append(f" [Figure: {alt}] " if alt else " [Figure] ")
        elif tag in ("br", "p", "div", "li", "tr", "td", "th"):
            table.cell.append(" ")
        elif tag == "table":
            self._nested += 1  # its text lands in the enclosing cell

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            if not self._foreign_depth and self.title.strip():
                self._title_seen = True
            return
        if tag in _SKIP:
            if tag not in _VOID and self._skip_depth:
                self._skip_depth -= 1
                if tag in _FOREIGN and self._foreign_depth:
                    self._foreign_depth -= 1
            return
        if self._skip_depth:
            return
        table = self._table
        if table is not None and table.caption is not None:
            if tag == "caption":
                text = " ".join("".join(table.caption).split())
                if text:
                    table.captions.append(text)
                table.caption = None
            return
        if table is not None and table.cell is not None:
            if self._nested:
                if tag == "table":
                    self._nested -= 1
                elif tag in ("td", "th"):
                    table.cell.append(" ")
                return
            if tag in ("td", "th"):
                self._close_cell()
                return
            if tag not in ("tr", "table") and tag not in _SECTIONS:
                return
        if tag in _HEADINGS and self._heading is not None:
            text = self._collected()
            if text:
                self._emit(Heading(provenance=self.provenance, content=text, level=self._heading))
            self._heading = None
        elif tag == "pre" and self._pre_depth:
            content = "".join(self._text).strip("\n")
            self._text.clear()
            self._pre_depth -= 1
            if content.strip():
                self._emit(
                    CodeBlock(provenance=self.provenance, content=content, language=self._pre_lang)
                )
        elif tag == "li":
            self._flush_paragraph()
        elif tag in ("ul", "ol") and self._lists:
            self._flush_paragraph()
            self._lists.pop()
            if not self._lists:
                self._emit_list()
        elif tag == "tr" and table is not None or tag in _SECTIONS and table is not None:
            self._end_row()
        elif tag == "table" and table is not None:
            self._end_table()
        elif tag in _BREAKS:
            self._flush_paragraph()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        table = self._table
        if table is not None and table.caption is not None:
            table.caption.append(data)
        elif table is not None and table.cell is not None:
            table.cell.append(data)
        elif table is not None:
            return  # whitespace between rows/cells
        else:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        if self._tables:
            self.warnings.append(
                ParseWarning(
                    code="unclosed_table",
                    message="The input ended inside a table (a cell or the table was never "
                    "closed); the table was closed at the end of the document.",
                )
            )
            while self._tables:
                self._end_table()
        self._flush_paragraph()
        while self._lists:  # unclosed lists at end of input
            self._lists.pop()
        self._emit_list()


class HTMLParser:
    name = "html"
    media_types: tuple[str, ...] = ("text/html", "application/xhtml+xml")
    suffixes: tuple[str, ...] = (".html", ".htm", ".xhtml")

    def parse(self, source: DocumentSource) -> ParsedDocument:
        text, warnings = decode_text(source.data, charset=source.charset, html=True)
        walker = _Walker(Provenance(source.name, 1))
        walker.feed(text)
        walker.close()
        metadata = {"title": " ".join(walker.title.split())} if walker.title.strip() else {}
        return ParsedDocument(
            source=source.name,
            media_type="text/html",
            pages=(Page(number=1, blocks=tuple(walker.blocks)),),
            warnings=tuple(warnings + walker.warnings),
            metadata=metadata,
        )
