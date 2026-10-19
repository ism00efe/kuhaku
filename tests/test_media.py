"""ImagePart / DocumentPart: construction, validation, and what they reveal in logs."""

from __future__ import annotations

import base64
import struct
import zlib

import pytest

from kuhaku import DocumentPart, ImagePart, Message, TextPart
from kuhaku.core.media import DEFAULT_MAX_IMAGE_BYTES, sniff_media_type


def tiny_png() -> bytes:
    """A valid 1x1 greyscale PNG, built by hand."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00"))
        + chunk(b"IEND", b"")
    )


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PDF = b"%PDF-1.4\n%fake but signed\n"


def test_from_bytes_infers_the_media_type():
    assert ImagePart.from_bytes(tiny_png()).media_type == "image/png"
    assert ImagePart.from_bytes(JPEG).media_type == "image/jpeg"
    assert DocumentPart.from_bytes(PDF).media_type == "application/pdf"
    assert ImagePart.from_bytes(JPEG, "image/jpg").media_type == "image/jpeg"


def test_sniff_covers_every_supported_type():
    assert sniff_media_type(b"GIF89a....") == "image/gif"
    assert sniff_media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_media_type(b"hello") is None


def test_a_mislabelled_image_is_rejected():
    with pytest.raises(ValueError, match="labelled image/jpeg but its bytes are image/png"):
        ImagePart.from_bytes(tiny_png(), "image/jpeg")


def test_unsupported_and_unknown_content_is_rejected():
    with pytest.raises(ValueError, match="Unsupported image media type"):
        ImagePart.from_bytes(PDF)
    with pytest.raises(ValueError, match="media type"):
        ImagePart.from_bytes(b"not an image at all")
    with pytest.raises(ValueError, match="do not look like image/png"):
        ImagePart.from_bytes(b"not an image at all", "image/png")
    with pytest.raises(ValueError, match="empty"):
        ImagePart(data=b"")
    with pytest.raises(ValueError, match="exactly one"):
        ImagePart(data=JPEG, url="https://example.com/a.jpg")
    with pytest.raises(TypeError, match="from_base64"):
        ImagePart(data="aGVsbG8=")  # type: ignore[arg-type]


def test_size_limit_is_configurable():
    big = JPEG + b"\x00" * 100
    with pytest.raises(ValueError, match="limit"):
        ImagePart.from_bytes(big, max_bytes=50)
    assert ImagePart.from_bytes(big, max_bytes=500).size == len(big)
    assert ImagePart.from_bytes(big).limit == DEFAULT_MAX_IMAGE_BYTES
    # The limit is not part of the part's identity.
    assert ImagePart.from_bytes(big, max_bytes=500) == ImagePart.from_bytes(big)


def test_base64_and_data_url_round_trip():
    part = ImagePart.from_bytes(tiny_png(), detail="low")
    assert base64.b64decode(part.base64) == part.data
    assert part.data_url.startswith("data:image/png;base64,")
    assert ImagePart.from_base64(part.base64, detail="low") == part
    assert ImagePart.from_url(part.data_url, detail="low") == part
    with pytest.raises(ValueError, match="Invalid base64"):
        ImagePart.from_base64("not base64!!")


def test_url_parts():
    part = ImagePart.from_url("https://example.com/photo.JPG?sig=secret")
    assert part.media_type == "image/jpeg" and part.data is None and part.size is None
    assert DocumentPart.from_url("gs://bucket/invoice.pdf").media_type == "application/pdf"
    with pytest.raises(ValueError, match="media type"):
        ImagePart.from_url("https://example.com/photo")
    assert ImagePart.from_url("https://example.com/photo", "image/webp").media_type == "image/webp"
    with pytest.raises(ValueError, match="from_path"):
        ImagePart.from_url("file:///etc/photo.png")
    with pytest.raises(ValueError, match="from_url"):
        ImagePart(url="data:image/png;base64,AAAA")
    with pytest.raises(ValueError, match="URL"):
        _ = ImagePart.from_url("https://example.com/a.png").base64


def test_from_path_reads_once_and_names_documents(tmp_path):
    image = tmp_path / "order.png"
    image.write_bytes(tiny_png())
    part = ImagePart.from_path(image)
    image.unlink()  # the part is self-contained
    assert part.media_type == "image/png" and part.data == tiny_png()

    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(PDF)
    document = DocumentPart.from_path(pdf)
    assert document.filename == "invoice.pdf"
    assert DocumentPart.from_path(pdf, filename="x.pdf").filename == "x.pdf"


def test_from_path_checks_the_size_before_reading(tmp_path):
    image = tmp_path / "huge.jpg"
    image.write_bytes(JPEG + b"\x00" * 1000)
    with pytest.raises(ValueError, match="huge.jpg"):
        ImagePart.from_path(image, max_bytes=100)


def test_detail_is_validated():
    with pytest.raises(ValueError, match="detail"):
        ImagePart.from_bytes(JPEG, detail="ultra")


def test_repr_and_describe_never_show_the_content():
    part = ImagePart.from_bytes(tiny_png())
    for shown in (repr(part), repr(Message.user([part])), str(part.describe())):
        assert part.base64 not in shown and "\\x89PNG" not in shown
    assert part.describe() == {"type": "image", "media_type": "image/png", "size": len(tiny_png())}
    url_part = ImagePart.from_url("https://example.com/a.png?token=secret")
    assert "secret" not in repr(url_part) and "secret" not in str(url_part.describe())


def test_fingerprint_is_stable_and_content_addressed():
    a = ImagePart.from_bytes(tiny_png())
    assert a.fingerprint == ImagePart.from_bytes(tiny_png()).fingerprint
    assert len(a.fingerprint) == 64
    assert a.fingerprint != ImagePart.from_bytes(tiny_png(), detail="high").fingerprint
    assert a.fingerprint != DocumentPart.from_bytes(PDF).fingerprint
    assert TextPart("x").fingerprint == TextPart("x").fingerprint != TextPart("y").fingerprint
    assert hash(a) == hash(ImagePart.from_bytes(tiny_png()))


def test_messages_carry_media_parts():
    part = ImagePart.from_bytes(tiny_png())
    message = Message.user([part, TextPart("what is this?")])
    assert message.parts == (part, TextPart("what is this?"))
    assert message.text == "what is this?"
