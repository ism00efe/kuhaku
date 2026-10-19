"""Image and document content parts: what a :class:`~kuhaku.core.models.Message` carries
besides text.

A part holds its content either inline (``data``, raw bytes) or by reference (``url``).
A file path is read once, at construction (:meth:`MediaPart.from_path`), so a part is a
self-contained value: comparing, hashing or fingerprinting it never touches the disk,
and what was validated is what is sent.

Validation happens at construction, never at send time: the media type must be one of
the part's supported types and -- for inline bytes -- must match the bytes' own
signature (a JPEG labelled ``image/png`` is rejected, as several APIs would reject it,
only later and less clearly); the size must be within ``max_bytes``. Whether a given
*provider* (or model) accepts the part is a separate question, answered by its
``capabilities.content_types`` when the message is sent.

Bytes and base64 never appear in a part's ``repr`` or in :meth:`MediaPart.describe`,
which is what kuhaku logs: media type, size, and whether the content is inline or a URL.
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self
from urllib.parse import urlparse

from .models import ContentPart

IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
"""The image types every built-in provider that accepts images accepts."""

DOCUMENT_MEDIA_TYPES = frozenset({"application/pdf"})

DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_DOCUMENT_BYTES = 32 * 1024 * 1024

IMAGE_DETAILS = frozenset({"low", "high", "auto"})

URL_SCHEMES = frozenset({"http", "https", "gs"})
"""``gs://`` is for Vertex AI, which reads Cloud Storage objects itself."""

# Aliases seen in the wild (file extensions, older libraries) for the canonical types.
_ALIASES = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg", "application/x-pdf": "application/pdf"}


def sniff_media_type(data: bytes) -> str | None:
    """The media type ``data`` starts like, among the supported ones, else ``None``."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    return None


def _normalize_media_type(media_type: str) -> str:
    media_type = media_type.split(";", 1)[0].strip().lower()
    return _ALIASES.get(media_type, media_type)


def _guess_from_name(name: str) -> str:
    guessed, _ = mimetypes.guess_type(name)
    return _normalize_media_type(guessed) if guessed else ""


def _megabytes(size: int) -> str:
    if size < 1024 * 1024:
        return f"{size} bytes"
    return f"{size / (1024 * 1024):.1f} MB"


@dataclass(frozen=True, kw_only=True, repr=False)
class MediaPart(ContentPart):
    """Base for binary content parts. Use a subclass: :class:`ImagePart`,
    :class:`DocumentPart`.

    Exactly one of ``data`` (raw bytes, not base64) and ``url`` is set. ``media_type``
    may be left empty when it can be inferred -- from the bytes' signature, or from the
    URL's file extension. ``max_bytes`` overrides the subclass's default size limit; it
    is not part of the part's identity (equality, fingerprint).
    """

    data: bytes | None = None
    url: str | None = None
    media_type: str = ""
    max_bytes: int | None = field(default=None, compare=False)

    media_types: ClassVar[frozenset[str]] = frozenset()
    default_max_bytes: ClassVar[int] = 0
    noun: ClassVar[str] = "content"

    def __post_init__(self) -> None:
        if (self.data is None) == (self.url is None):
            raise ValueError(f"{type(self).__name__} needs exactly one of data= or url=.")
        media_type = _normalize_media_type(self.media_type)
        if self.data is not None:
            data = self.data
            if isinstance(data, (bytearray, memoryview)):
                data = bytes(data)
                object.__setattr__(self, "data", data)
            if not isinstance(data, bytes):
                raise TypeError(
                    f"{type(self).__name__}.data must be bytes, got {type(data).__name__} "
                    f"(for base64 text use {type(self).__name__}.from_base64)."
                )
            if not data:
                raise ValueError(f"{type(self).__name__}.data is empty.")
            limit = self.limit
            if len(data) > limit:
                raise ValueError(
                    f"The {self.noun} is {_megabytes(len(data))}; the limit is "
                    f"{_megabytes(limit)}. Pass max_bytes= to raise it."
                )
            sniffed = sniff_media_type(data)
            if not media_type:
                media_type = sniffed or ""
            elif sniffed is not None and sniffed != media_type:
                raise ValueError(
                    f"The {self.noun} was labelled {media_type} but its bytes are {sniffed}."
                )
            elif sniffed is None and media_type in self.media_types:
                raise ValueError(f"The {self.noun}'s bytes do not look like {media_type}.")
        else:
            url = self.url
            if not isinstance(url, str):
                raise TypeError(f"{type(self).__name__}.url must be a str.")
            scheme = urlparse(url).scheme.lower()
            if scheme == "data":
                raise ValueError(
                    f"A data: URL is inline content; use {type(self).__name__}.from_url, "
                    f"which decodes it."
                )
            if scheme not in URL_SCHEMES:
                raise ValueError(
                    f"{type(self).__name__}.url must be a {'/'.join(sorted(URL_SCHEMES))} "
                    f"URL; for a local file use {type(self).__name__}.from_path."
                )
            if not media_type:
                media_type = _guess_from_name(urlparse(url).path)
        if not media_type:
            raise ValueError(
                f"Could not tell the {self.noun}'s media type; pass media_type= "
                f"(one of {', '.join(sorted(self.media_types))})."
            )
        if media_type not in self.media_types:
            raise ValueError(
                f"Unsupported {self.noun} media type '{media_type}'; supported: "
                f"{', '.join(sorted(self.media_types))}."
            )
        object.__setattr__(self, "media_type", media_type)

    # --- constructors -----------------------------------------------------------------

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str | None = None, **kwargs: Any) -> Self:
        """From raw bytes; ``media_type`` is inferred from the bytes when omitted."""

        return cls(data=data, media_type=media_type or "", **kwargs)

    @classmethod
    def from_base64(cls, text: str, media_type: str | None = None, **kwargs: Any) -> Self:
        """From base64 text (whitespace and line breaks are tolerated)."""

        try:
            data = base64.b64decode("".join(text.split()), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid base64 {cls.noun}: {exc}") from None
        return cls(data=data, media_type=media_type or "", **kwargs)

    @classmethod
    def from_path(cls, path: str | Path, media_type: str | None = None, **kwargs: Any) -> Self:
        """Reads the file now; ``media_type`` comes from its bytes, else its extension.

        The size limit is checked before the file is read.
        """

        path = Path(path)
        limit = kwargs.get("max_bytes") or cls.default_max_bytes
        size = path.stat().st_size
        if size > limit:
            raise ValueError(
                f"{path.name} is {_megabytes(size)}; the {cls.noun} limit is "
                f"{_megabytes(limit)}. Pass max_bytes= to raise it."
            )
        data = path.read_bytes()
        if not media_type and sniff_media_type(data) is None:
            media_type = _guess_from_name(path.name)
        return cls._from_file(data, media_type or "", path, **kwargs)

    @classmethod
    def _from_file(cls, data: bytes, media_type: str, path: Path, **kwargs: Any) -> Self:
        return cls(data=data, media_type=media_type, **kwargs)

    @classmethod
    def from_url(cls, url: str, media_type: str | None = None, **kwargs: Any) -> Self:
        """By reference (http/https/gs). A ``data:`` URL is decoded into inline bytes.

        Nothing is downloaded: the provider fetches the URL, if it accepts URLs at all.
        """

        if url[:5].lower() == "data:":
            header, sep, payload = url[5:].partition(",")
            if not sep or not header.lower().endswith(";base64"):
                raise ValueError("Only base64 data: URLs are supported.")
            declared = header[: -len(";base64")]
            return cls.from_base64(payload, media_type or declared or None, **kwargs)
        return cls(url=url, media_type=media_type or "", **kwargs)

    # --- reading ----------------------------------------------------------------------

    @property
    def limit(self) -> int:
        return self.max_bytes or self.default_max_bytes

    @property
    def size(self) -> int | None:
        """Bytes held inline; ``None`` for a URL part."""

        return len(self.data) if self.data is not None else None

    @property
    def base64(self) -> str:
        if self.data is None:
            raise ValueError(f"This {self.noun} is a URL, not inline bytes.")
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.base64}"

    def describe(self) -> dict[str, Any]:
        """What may be logged about this part: never its bytes, base64 or URL (a URL
        can carry a signed token)."""

        described: dict[str, Any] = {"type": self.type, "media_type": self.media_type}
        if self.data is not None:
            described["size"] = len(self.data)
        else:
            described["source"] = "url"
        return described

    def __repr__(self) -> str:
        source = f"size={len(self.data)}" if self.data is not None else "url=..."
        return (
            f"{type(self).__name__}(media_type={self.media_type!r}, {source}, "
            f"sha256={self.fingerprint[:12]}...)"
        )


@dataclass(frozen=True, kw_only=True, repr=False)
class ImagePart(MediaPart):
    """An image: PNG, JPEG, GIF or WebP, up to 20 MB by default.

    ``detail`` (``"low"``/``"high"``/``"auto"``) is a resolution hint only OpenAI-style
    APIs read; the other providers ignore it.
    """

    detail: str | None = None

    type: ClassVar[str] = "image"
    media_types: ClassVar[frozenset[str]] = IMAGE_MEDIA_TYPES
    default_max_bytes: ClassVar[int] = DEFAULT_MAX_IMAGE_BYTES
    noun: ClassVar[str] = "image"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.detail is not None and self.detail not in IMAGE_DETAILS:
            raise ValueError(
                f"ImagePart.detail must be one of {', '.join(sorted(IMAGE_DETAILS))}; "
                f"got '{self.detail}'."
            )


@dataclass(frozen=True, kw_only=True, repr=False)
class DocumentPart(MediaPart):
    """A document: PDF, up to 32 MB by default.

    ``filename`` is sent where an API wants one (OpenAI's file input, Anthropic's
    document title); :meth:`from_path` fills it from the path.
    """

    filename: str | None = None

    type: ClassVar[str] = "document"
    media_types: ClassVar[frozenset[str]] = DOCUMENT_MEDIA_TYPES
    default_max_bytes: ClassVar[int] = DEFAULT_MAX_DOCUMENT_BYTES
    noun: ClassVar[str] = "document"

    @classmethod
    def _from_file(cls, data: bytes, media_type: str, path: Path, **kwargs: Any) -> Self:
        kwargs.setdefault("filename", path.name)
        return cls(data=data, media_type=media_type, **kwargs)


__all__ = [
    "DEFAULT_MAX_DOCUMENT_BYTES",
    "DEFAULT_MAX_IMAGE_BYTES",
    "DOCUMENT_MEDIA_TYPES",
    "IMAGE_DETAILS",
    "IMAGE_MEDIA_TYPES",
    "URL_SCHEMES",
    "DocumentPart",
    "ImagePart",
    "MediaPart",
    "sniff_media_type",
]
