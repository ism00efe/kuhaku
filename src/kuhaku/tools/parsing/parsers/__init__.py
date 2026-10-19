"""Built-in parsers. Each imports its optional backend only inside ``parse``."""

from .docx import DocxParser
from .html import HTMLParser
from .image import ImageParser
from .pdf import PdfParser
from .text import MarkdownParser, TextParser

__all__ = [
    "DocxParser",
    "HTMLParser",
    "ImageParser",
    "MarkdownParser",
    "PdfParser",
    "TextParser",
]
