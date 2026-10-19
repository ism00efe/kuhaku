"""Images (PNG, JPEG, TIFF, ...), read through an OCR engine.

Without an engine the image still parses -- to a single figure placeholder plus an
``ocr_unavailable`` warning -- so a batch over mixed files does not stop at the first
picture, and the caller can see exactly which files were not read.

A multi-page TIFF (a scanned fax or document) becomes one page per frame: each frame is
re-encoded as PNG (Pillow, the ``parsing-ocr`` extra) and recognized on its own, so
page numbers in provenance match the TIFF's. ``DocumentSource.max_pages`` limits how
many frames are read.
"""

from __future__ import annotations

import io
import logging

from ..document import (
    Figure,
    Page,
    ParsedDocument,
    ParseWarning,
    Provenance,
    page_limit_warning,
)
from ..errors import is_installed, require
from ..ocr import OCR_EXTRA, OCREngine, ocr_unavailable_hint
from ..registry import DocumentSource
from .text import paragraphs

logger = logging.getLogger(__name__)

_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")


class ImageParser:
    name = "image"
    media_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/bmp",
        "image/gif",
        "image/webp",
    )
    suffixes: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
    )

    def __init__(self, *, ocr: OCREngine | None = None) -> None:
        self.ocr = ocr

    def parse(self, source: DocumentSource) -> ParsedDocument:
        media_type = source.media_type or self.media_types[0]
        if self.ocr is None:
            page = Page(
                number=1, blocks=(Figure(provenance=Provenance(source.name, 1)),)
            )
            warning = ParseWarning(
                code="ocr_unavailable",
                message=f"'{source.name}' is an image and no OCR engine is available: "
                f"{ocr_unavailable_hint()}.",
                page=1,
            )
            return ParsedDocument(
                source=source.name,
                media_type=media_type,
                pages=(page,),
                warnings=(warning,),
            )

        frames, total = _frames(source)
        warnings: list[ParseWarning] = []
        if source.max_pages is not None and total > source.max_pages:
            warnings.append(page_limit_warning(total, source.max_pages))
        pages: list[Page] = []
        for number, frame in enumerate(frames, start=1):
            text = self.ocr.recognize(frame)
            blocks = paragraphs(text, Provenance(source.name, number, ocr=True))
            if not blocks:
                where = (
                    f"page {number} of '{source.name}'"
                    if total > 1
                    else f"'{source.name}'"
                )
                warnings.append(
                    ParseWarning(
                        code="ocr_empty",
                        message=f"OCR found no text in {where}.",
                        page=number,
                    )
                )
            pages.append(Page(number=number, blocks=tuple(blocks), ocr=True))
        return ParsedDocument(
            source=source.name,
            media_type=media_type,
            pages=tuple(pages),
            warnings=tuple(warnings),
            metadata={"ocr_engine": self.ocr.name},
        )


def _frames(source: DocumentSource) -> tuple[list[bytes], int]:
    """``(frames to recognize, total frame count)``. Only a TIFF with several frames is
    split (and only when Pillow is installed); anything else is one frame, as given."""

    if not source.data.startswith(_TIFF_MAGIC) or not is_installed("PIL.Image"):
        return [source.data], 1
    pil_image = require("PIL.Image", extra=OCR_EXTRA, purpose="Reading multi-page TIFF")
    try:
        with pil_image.open(io.BytesIO(source.data)) as image:
            total = int(getattr(image, "n_frames", 1))
            if total <= 1:
                return [source.data], 1
            limit = total if source.max_pages is None else min(total, source.max_pages)
            frames: list[bytes] = []
            for index in range(limit):
                image.seek(index)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                frames.append(buffer.getvalue())
            return frames, total
    except (OSError, ValueError, EOFError):
        # Pillow could not split it; the OCR engine may still read the first frame.
        logger.debug("could not split TIFF frames", exc_info=True)
        return [source.data], 1
