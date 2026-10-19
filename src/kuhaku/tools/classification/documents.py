"""A structural contract for "a document with pages": what :class:`Classifier` accepts
besides plain text.

``kuhaku.tools.parsing.ParsedDocument`` satisfies :class:`Document` without either tool
importing the other -- the protocol is structural.

A document is a ``source`` name plus ``pages``, each with a 1-based ``number`` and its
``text`` (and, optionally, ``to_markdown()`` -- headings and tables keep their shape,
which is what an LLM reads best). :func:`document_text` flattens one into a single
string and remembers which page each part came from.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["Document", "DocumentPage", "DocumentText", "document_text", "is_document"]

PAGE_SEPARATOR = "\n\n"


@runtime_checkable
class DocumentPage(Protocol):
    @property
    def number(self) -> int: ...

    @property
    def text(self) -> str: ...


@runtime_checkable
class Document(Protocol):
    @property
    def source(self) -> str: ...

    @property
    def pages(self) -> Iterable[DocumentPage]: ...


@dataclass(frozen=True)
class DocumentText:
    """A document flattened to one string. ``pages`` holds ``(page number, text)`` for
    every non-empty page, in order, as joined into ``text``."""

    source: str
    text: str
    pages: tuple[tuple[int, str], ...] = ()


def is_document(value: object) -> bool:
    return not isinstance(value, str | bytes) and isinstance(value, Document)


def document_text(document: Document, *, markdown: bool = True) -> DocumentText:
    """Flatten ``document``: each page's Markdown (when the page has ``to_markdown``
    and ``markdown`` is true) or text, joined by blank lines; empty pages are skipped."""

    pages: list[tuple[int, str]] = []
    for page in document.pages:
        render = getattr(page, "to_markdown", None) if markdown else None
        body = render() if callable(render) else page.text
        if body and body.strip():
            pages.append((int(page.number), body))
    return DocumentText(
        source=str(document.source),
        text=PAGE_SEPARATOR.join(body for _, body in pages),
        pages=tuple(pages),
    )
