"""Fixtures for the parsing tests: tiny documents generated in memory, and a fake OCR.

The PDFs are written by hand (a few objects and an xref table) so the tests need no PDF
authoring library and nothing is committed as a binary fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class PdfPage:
    texts: list[tuple[float, float, float, str]] = field(default_factory=list)
    """(x, y, font size, text); y is measured from the bottom, as PDF does."""
    lines: list[tuple[float, float, float, float]] = field(default_factory=list)
    image: tuple[int, int] | None = None
    """(width, height) of a grey image drawn over the whole page -- a "scan"."""


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[PdfPage], *, title: str | None = None) -> bytes:
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # filled in once the page tree number is known
    pages_obj = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    for page in pages:
        ops: list[str] = []
        for x0, y0, x1, y1 in page.lines:
            ops.append(f"{x0} {y0} m {x1} {y1} l S")
        for x, y, size, text in page.texts:
            ops.append(f"BT /F1 {size} Tf {x} {y} Td ({_escape(text)}) Tj ET")
        resources = f"/Font << /F1 {font} 0 R >>"
        if page.image is not None:
            w, h = page.image
            pixels = bytes((x * 7 + y * 3) % 256 for y in range(h) for x in range(w))
            image = add(
                f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace "
                f"/DeviceGray /BitsPerComponent 8 /Length {len(pixels)} >>\nstream\n".encode()
                + pixels
                + b"\nendstream"
            )
            resources += f" /XObject << /Im1 {image} 0 R >>"
            ops.append("q 612 0 0 792 0 0 cm /Im1 Do Q")
        stream = "\n".join(ops).encode("latin-1")
        content = add(
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << {resources} >> /Contents {content} 0 R >>".encode()
            )
        )
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_obj - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    info = add(f"<< /Title ({_escape(title)}) >>".encode()) if title else None

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = f"/Size {len(objects) + 1} /Root {catalog} 0 R"
    if info:
        trailer += f" /Info {info} 0 R"
    out += f"trailer\n<< {trailer} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def table_page() -> PdfPage:
    """A heading, two body paragraphs, a bullet list and a ruled 2x3 table."""

    lines = [(72, y, 472, y) for y in (600, 580, 560, 540)]
    lines += [(x, 540, x, 600) for x in (72, 272, 472)]
    return PdfPage(
        texts=[
            (72, 720, 20, "Quarterly Report"),
            (72, 690, 11, "Revenue grew in every region this quarter, led by exports."),
            (72, 676, 11, "Costs stayed flat across the same period of time."),
            (72, 650, 11, "A second paragraph starts after a visible vertical gap."),
            (72, 625, 11, "- first point"),
            (72, 612, 11, "- second point"),
            (80, 585, 11, "Region"),
            (280, 585, 11, "Sales"),
            (80, 565, 11, "North"),
            (280, 565, 11, "120"),
            (80, 545, 11, "South"),
            (280, 545, 11, "95"),
        ],
        lines=lines,
    )


class FakeOCR:
    name = "fake-ocr"

    def __init__(self, text: str = "Recognized line one.\n\nRecognized line two.") -> None:
        self.text = text
        self.calls: list[bytes] = []

    def recognize(self, image: bytes) -> str:
        self.calls.append(image)
        return self.text


class FailingOCR:
    name = "failing-ocr"

    def recognize(self, image: bytes) -> str:
        raise RuntimeError("engine crashed")


@pytest.fixture
def fake_ocr() -> FakeOCR:
    return FakeOCR()
