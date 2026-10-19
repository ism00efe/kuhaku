"""The :class:`Parser` protocol and the registry that picks one per file.

A parser declares the media types and file suffixes it handles; the registry maps both
to it. Selection order for a source is: an explicit media type, then the file suffix,
then the content's magic bytes (``filetype``, already a kuhaku dependency), then -- for
content that looks like text (a BOM, valid UTF-8, or no NUL bytes) -- plain text,
decoded by :mod:`~kuhaku.tools.parsing.encoding`. Registering a parser for a type that is
already taken replaces the earlier one, which is how a caller swaps in their own.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Protocol, runtime_checkable

import filetype

from .document import ParsedDocument
from .encoding import looks_like_text
from .errors import UnsupportedFormat


@dataclass(frozen=True)
class DocumentSource:
    """The raw content handed to a parser, plus the name provenance will cite."""

    data: bytes
    name: str
    media_type: str = ""
    charset: str = ""
    """The ``charset`` parameter of the media type the caller declared, if any --
    text parsers decode with it before guessing."""
    password: str | None = field(default=None, repr=False)
    """The password of an encrypted document (PDF)."""
    max_pages: int | None = None
    """Read at most this many pages (PDF pages, TIFF frames); ``None`` for all."""

    @property
    def suffix(self) -> str:
        return PurePath(self.name).suffix.lower()


@runtime_checkable
class Parser(Protocol):
    name: str
    media_types: tuple[str, ...]
    suffixes: tuple[str, ...]

    def parse(self, source: DocumentSource) -> ParsedDocument: ...


class ParserRegistry:
    def __init__(self, parsers: Iterable[Parser] = ()) -> None:
        self._by_media_type: dict[str, Parser] = {}
        self._by_suffix: dict[str, Parser] = {}
        for parser in parsers:
            self.register(parser)

    def register(self, parser: Parser) -> None:
        for media_type in parser.media_types:
            self._by_media_type[media_type.lower()] = parser
        for suffix in parser.suffixes:
            self._by_suffix[suffix.lower()] = parser

    @property
    def media_types(self) -> list[str]:
        return sorted(self._by_media_type)

    @property
    def suffixes(self) -> list[str]:
        return sorted(self._by_suffix)

    def select(self, source: DocumentSource) -> tuple[Parser, str]:
        """Return the parser for ``source`` and the media type it was selected for."""

        if source.media_type:
            media_type = _base_media_type(source.media_type)
            if media_type in self._by_media_type:
                return self._by_media_type[media_type], media_type
        if source.suffix in self._by_suffix:
            parser = self._by_suffix[source.suffix]
            return parser, _media_type_for(parser, source)
        kind = filetype.guess(source.data) if source.data else None
        if kind is not None and kind.mime in self._by_media_type:
            return self._by_media_type[kind.mime], kind.mime
        if kind is None and "text/plain" in self._by_media_type and looks_like_text(source.data):
            return self._by_media_type["text/plain"], "text/plain"
        detected = kind.mime if kind is not None else (source.media_type or "unknown")
        raise UnsupportedFormat(
            f"No parser for '{source.name}' (detected type: {detected}). Supported "
            f"suffixes: {', '.join(self.suffixes)}."
        )


def _base_media_type(media_type: str) -> str:
    return media_type.split(";", 1)[0].strip().lower()


def _media_type_for(parser: Parser, source: DocumentSource) -> str:
    guessed = filetype.guess(source.data) if source.data else None
    if guessed is not None and guessed.mime in parser.media_types:
        return guessed.mime
    return parser.media_types[0] if parser.media_types else "application/octet-stream"
