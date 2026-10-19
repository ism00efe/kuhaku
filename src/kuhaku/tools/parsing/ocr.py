"""Optical character recognition: the :class:`OCREngine` protocol and a Tesseract engine.

An engine takes encoded image bytes (PNG, JPEG, TIFF, ...) and returns the recognized
text. That is deliberately all it does, so any OCR backend -- a cloud API, RapidOCR, an
in-house model -- plugs in with one method.

``"auto"`` (see :func:`resolve_ocr`) picks :class:`TesseractOCR` only when both the
``parsing-ocr`` extra is installed and the ``tesseract`` binary is on ``PATH``; otherwise
it resolves to no engine, and parsers report pages they could not read as warnings.
"""

from __future__ import annotations

import io
import logging
import shutil
from typing import Literal, Protocol, runtime_checkable

from .errors import MissingDependency, ParsingError, is_installed, require

logger = logging.getLogger(__name__)

OCR_EXTRA = "parsing-ocr"


@runtime_checkable
class OCREngine(Protocol):
    name: str

    def recognize(self, image: bytes) -> str:
        """Return the text in ``image`` (encoded image bytes), or ``""`` if none."""
        ...


# Preferred languages for ``lang="auto"``, in order: every one installed is used.
AUTO_LANGUAGES = ("tur", "eng")


class TesseractOCR:
    """OCR through the Tesseract engine (``pytesseract`` + Pillow, and the system
    ``tesseract`` binary).

    ``lang`` is a Tesseract language string (``"eng"``, ``"tur+eng"``) or ``"auto"``
    (the default): every language of :data:`AUTO_LANGUAGES` that the installed
    Tesseract has, joined -- ``tur+eng`` when both are there -- falling back to
    Tesseract's own default when it has none of them. Resolved once, on first use.
    """

    name = "tesseract"

    def __init__(self, *, lang: str = "auto", config: str = "") -> None:
        self.lang = lang
        self.config = config
        self._resolved: str | None = None if lang == "auto" else lang

    def resolved_lang(self) -> str:
        """The language string ``recognize`` passes to Tesseract."""

        if self._resolved is None:
            pytesseract = require("pytesseract", extra=OCR_EXTRA, purpose="OCR")
            try:
                installed = set(pytesseract.get_languages(config=""))
            except Exception:
                logger.debug("could not list Tesseract languages", exc_info=True)
                installed = set()
            chosen = [code for code in AUTO_LANGUAGES if code in installed]
            self._resolved = "+".join(chosen) if chosen else "eng"
        return self._resolved

    def recognize(self, image: bytes) -> str:
        pytesseract = require("pytesseract", extra=OCR_EXTRA, purpose="OCR")
        pil_image = require("PIL.Image", extra=OCR_EXTRA, purpose="OCR")
        try:
            with pil_image.open(io.BytesIO(image)) as img:
                lang = self.resolved_lang()
                return str(pytesseract.image_to_string(img, lang=lang, config=self.config))
        except pytesseract.TesseractNotFoundError as exc:
            raise MissingDependency(
                "OCR needs the Tesseract binary, which was not found on PATH. Install "
                "Tesseract (https://tesseract-ocr.github.io/tessdoc/Installation.html) "
                "or pass a different OCREngine."
            ) from exc
        except pil_image.UnidentifiedImageError as exc:
            raise ParsingError(f"Not a readable image: {exc}") from exc


def tesseract_available() -> bool:
    """True when :class:`TesseractOCR` can run here -- checked without starting it."""

    return (
        is_installed("pytesseract")
        and is_installed("PIL.Image")
        and shutil.which("tesseract") is not None
    )


def resolve_ocr(
    ocr: OCREngine | None | Literal["auto"], *, lang: str = "auto"
) -> OCREngine | None:
    """``"auto"`` -> a :class:`TesseractOCR` (with ``lang``) when usable now, else
    ``None``. An explicit engine or ``None`` is returned unchanged."""

    if ocr != "auto":
        return ocr  # type: ignore[return-value]
    if tesseract_available():
        return TesseractOCR(lang=lang)
    logger.debug("No OCR engine available; scanned pages will be reported as warnings.")
    return None


def ocr_unavailable_hint() -> str:
    return (
        f'install the OCR extra (pip install "kuhaku[{OCR_EXTRA}]") and the Tesseract '
        "binary, or pass ocr=<an OCREngine>"
    )
