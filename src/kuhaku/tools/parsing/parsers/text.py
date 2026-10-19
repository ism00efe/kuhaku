"""Plain text and Markdown. No dependency beyond the standard library.

The Markdown parser is a line-based reader for the block structure that matters
downstream -- ATX headings, lists, pipe tables, fenced code, standalone images,
paragraphs. Inline markup (emphasis, links) is kept verbatim inside the block text.
"""

from __future__ import annotations

import re

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
from ..encoding import decode_text as _decode
from ..registry import DocumentSource


def decode_text(
    data: bytes, *, charset: str = "", html: bool = False
) -> tuple[str, list[ParseWarning]]:
    """Decode text of unknown encoding -- see :mod:`kuhaku.tools.parsing.encoding`."""

    return _decode(data, charset=charset, html=html)


def paragraphs(text: str, provenance: Provenance) -> list[Block]:
    """Split on blank lines; each run of non-blank lines is one paragraph."""

    parts = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    return [Paragraph(provenance=provenance, content=p.strip()) for p in parts if p.strip()]


class TextParser:
    name = "text"
    media_types: tuple[str, ...] = ("text/plain",)
    suffixes: tuple[str, ...] = (".txt", ".text", ".log")

    def parse(self, source: DocumentSource) -> ParsedDocument:
        text, warnings = decode_text(source.data, charset=source.charset)
        page = Page(number=1, blocks=tuple(paragraphs(text, Provenance(source.name, 1))))
        return ParsedDocument(
            source=source.name, media_type="text/plain", pages=(page,), warnings=tuple(warnings)
        )


_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*([\w+-]*)")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|?.*\|.*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_IMAGE_RE = re.compile(r"^\s*!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]*)(?:\s+\"[^\"]*\")?\)\s*$")


def _split_row(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    stripped = stripped.removeprefix("|")
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    cells = re.split(r"(?<!\\)\|", stripped)
    return tuple(cell.strip().replace("\\|", "|") for cell in cells)


def parse_markdown(text: str, provenance: Provenance) -> list[Block]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append(Paragraph(provenance=provenance, content="\n".join(para).strip()))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            flush_para()
            i += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            flush_para()
            marker = fence.group(1)
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(marker):
                body.append(lines[i])
                i += 1
            i += 1  # the closing fence (or past the end for an unclosed one)
            blocks.append(
                CodeBlock(provenance=provenance, content="\n".join(body), language=fence.group(2))
            )
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_para()
            blocks.append(
                Heading(
                    provenance=provenance,
                    content=heading.group(2).strip(),
                    level=len(heading.group(1)),
                )
            )
            i += 1
            continue

        if (
            not para
            and _TABLE_ROW_RE.match(line)
            and i + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[i + 1])
            and "|" in lines[i + 1]
        ):
            rows = [_split_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append(Table(provenance=provenance, rows=tuple(rows), has_header=True))
            continue

        image = _IMAGE_RE.match(line)
        if image:
            flush_para()
            blocks.append(Figure(provenance=provenance, alt=image["alt"], src=image["src"]))
            i += 1
            continue

        item = _LIST_RE.match(line)
        if item and not para:
            items: list[ListItem] = []
            ordered = item.group(2)[0].isdigit()
            base_indent = len(item.group(1))
            while i < len(lines):
                match = _LIST_RE.match(lines[i])
                if match:
                    level = max(0, (len(match.group(1)) - base_indent) // 2)
                    items.append(ListItem(text=match.group(3).strip(), level=level))
                elif lines[i].strip() and lines[i].startswith((" ", "\t")) and items:
                    # An indented continuation line belongs to the item above it.
                    last = items[-1]
                    items[-1] = ListItem(text=f"{last.text} {lines[i].strip()}", level=last.level)
                else:
                    break
                i += 1
            blocks.append(ListBlock(provenance=provenance, items=tuple(items), ordered=ordered))
            continue

        para.append(line)
        i += 1

    flush_para()
    return blocks


class MarkdownParser:
    name = "markdown"
    media_types: tuple[str, ...] = ("text/markdown", "text/x-markdown")
    suffixes: tuple[str, ...] = (".md", ".markdown")

    def parse(self, source: DocumentSource) -> ParsedDocument:
        text, warnings = decode_text(source.data, charset=source.charset)
        blocks = parse_markdown(text, Provenance(source.name, 1))
        return ParsedDocument(
            source=source.name,
            media_type="text/markdown",
            pages=(Page(number=1, blocks=tuple(blocks)),),
            warnings=tuple(warnings),
        )
