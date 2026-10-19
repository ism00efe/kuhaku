"""``DocumentParser``: the entry point -- one registry, one call per file."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Literal

from kuhaku.core.exceptions import ConfigError
from kuhaku.core.observability import bind_trace_id, instrumented_step

from .document import ParsedDocument, page_limit_warning
from .encoding import charset_from_media_type
from .errors import DocumentTooLarge
from .ocr import OCREngine, resolve_ocr
from .parsers import (
    DocxParser,
    HTMLParser,
    ImageParser,
    MarkdownParser,
    PdfParser,
    TextParser,
)
from .registry import DocumentSource, ParserRegistry

logger = logging.getLogger(__name__)


def default_registry(
    ocr: OCREngine | None | Literal["auto"] = "auto", *, ocr_lang: str = "auto"
) -> ParserRegistry:
    """Every built-in parser. ``ocr`` goes to the PDF and image parsers (see
    :func:`~kuhaku.tools.parsing.ocr.resolve_ocr` for ``"auto"``); ``ocr_lang`` is the
    language of the Tesseract engine ``"auto"`` builds (see
    :class:`~kuhaku.tools.parsing.ocr.TesseractOCR`)."""

    engine = resolve_ocr(ocr, lang=ocr_lang)
    return ParserRegistry(
        [
            TextParser(),
            MarkdownParser(),
            HTMLParser(),
            DocxParser(),
            PdfParser(ocr=engine),
            ImageParser(ocr=engine),
        ]
    )


class DocumentParser:
    """Parses a file into a :class:`~kuhaku.tools.parsing.document.ParsedDocument`.

    ``registry`` decides which parser handles which file type; the default holds every
    built-in parser. ``ocr`` configures that default registry only -- an OCR engine for
    a custom registry is passed to its parsers directly, so passing both is an error
    rather than a silently ignored argument. ``ocr_lang`` likewise configures the
    default registry's Tesseract engine: ``"auto"`` (the default) uses ``tur+eng`` when
    both are installed, else whichever of them is.

    ``max_bytes`` refuses larger content with :class:`DocumentTooLarge` before it is
    read or parsed; ``max_pages`` reads at most that many pages (PDF pages, TIFF
    frames) and reports a ``page_limit`` warning when there were more. Both default to
    ``None`` -- no limit.
    """

    def __init__(
        self,
        *,
        registry: ParserRegistry | None = None,
        ocr: OCREngine | None | Literal["auto"] = "auto",
        ocr_lang: str = "auto",
        max_bytes: int | None = None,
        max_pages: int | None = None,
    ) -> None:
        if registry is not None and (ocr != "auto" or ocr_lang != "auto"):
            raise ConfigError(
                "DocumentParser got both registry= and ocr=; ocr= only configures (as does "
                "ocr_lang=) the default registry. Pass the OCR engine to your registry's "
                "parsers instead."
            )
        for label, limit in (("max_bytes", max_bytes), ("max_pages", max_pages)):
            if limit is not None and limit < 1:
                raise ConfigError(
                    f"{label} must be a positive number or None; got {limit}."
                )
        self.registry = (
            registry
            if registry is not None
            else default_registry(ocr, ocr_lang=ocr_lang)
        )
        self.max_bytes = max_bytes
        self.max_pages = max_pages

    def parse(
        self,
        source: str | Path | bytes,
        *,
        filename: str | None = None,
        media_type: str | None = None,
        trace_id: str | None = None,
        password: str | None = None,
    ) -> ParsedDocument:
        """Parse a file path, or raw bytes.

        A ``str`` is a path, never content: to parse text you already hold, pass
        ``text.encode()`` with a ``filename`` whose suffix names the format.
        ``filename`` is the name provenance cites (defaults to the path's file name, or
        ``"document"`` for bytes) and, with ``media_type``, selects the parser -- see
        :class:`~kuhaku.tools.parsing.registry.ParserRegistry`.

        ``trace_id`` binds this call's log records and span to an id; without one, an
        id bound by the caller is inherited, else a new one is minted.

        ``password`` opens an encrypted PDF; without it (or with a wrong one) an
        encrypted PDF raises :class:`ParsingError`.
        """

        if isinstance(source, bytes | bytearray):
            data = bytes(source)
            name = filename or "document"
        elif isinstance(source, str | Path):
            path = Path(source)
            if self.max_bytes is not None and path.stat().st_size > self.max_bytes:
                raise DocumentTooLarge(
                    f"'{path.name}' is {path.stat().st_size} bytes; max_bytes is {self.max_bytes}."
                )
            data = path.read_bytes()
            name = filename or path.name
        else:
            raise TypeError(
                f"source must be a path or bytes, got {type(source).__name__}."
            )
        if self.max_bytes is not None and len(data) > self.max_bytes:
            raise DocumentTooLarge(
                f"'{name}' is {len(data)} bytes; max_bytes is {self.max_bytes}."
            )

        with (
            bind_trace_id(trace_id) as tid,
            instrumented_step("parsing", logger, size_bytes=len(data)) as rec,
        ):
            doc_source = DocumentSource(
                data=data,
                name=name,
                media_type=media_type or "",
                charset=charset_from_media_type(media_type or ""),
                password=password,
                max_pages=self.max_pages,
            )
            parser, selected = self.registry.select(doc_source)
            rec.set(parser=parser.name, media_type=selected)
            doc = parser.parse(dataclasses.replace(doc_source, media_type=selected))
            doc = dataclasses.replace(doc, trace_id=tid)
            doc = _limit_pages(doc, self.max_pages)
            rec.set(
                page_count=len(doc.pages),
                block_count=sum(len(page.blocks) for page in doc.pages),
                table_count=len(doc.tables),
                ocr_page_count=len(doc.ocr_pages),
                warning_count=len(doc.warnings),
            )
        if doc.warnings:
            logger.warning(
                "Parsed a %s document with %d warning(s): %s",
                selected,
                len(doc.warnings),
                ", ".join(sorted({w.code for w in doc.warnings})),
            )
        return doc


def _limit_pages(doc: ParsedDocument, max_pages: int | None) -> ParsedDocument:
    """Cut ``doc`` to ``max_pages`` -- the safety net for a parser that does not honour
    ``DocumentSource.max_pages`` itself (the built-in PDF and image parsers do, and add
    the ``page_limit`` warning)."""

    if max_pages is None or len(doc.pages) <= max_pages:
        return doc
    warnings = doc.warnings
    if not any(w.code == "page_limit" for w in warnings):
        warnings = (*warnings, page_limit_warning(len(doc.pages), max_pages))
    return dataclasses.replace(doc, pages=doc.pages[:max_pages], warnings=warnings)
