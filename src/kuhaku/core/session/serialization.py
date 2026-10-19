"""The stored form of a session's messages, tool state and metadata.

Versioned JSON. A message is ``{"role", "metadata"}`` plus either ``"content"`` (a
plain string) or ``"parts"`` (a list of ``{"type": ..., ...}`` objects) -- the same
two forms :class:`~kuhaku.core.models.Message` accepts, so a round trip gives back an
equal message.

Content parts go through a codec registry keyed by the part's ``type``. Only
:class:`~kuhaku.core.models.TextPart`, the tool-calling parts
(:class:`~kuhaku.core.models.ToolCallPart`, :class:`~kuhaku.core.models.ToolResultPart`)
and the media parts (:class:`~kuhaku.core.media.ImagePart`,
:class:`~kuhaku.core.media.DocumentPart` -- inline bytes stored as base64) are
registered here; a new part type registers its own codec with
:func:`register_part_codec` beside its definition. A stored part whose type has no codec is an error, never
skipped: a session that silently lost an image or a tool result would replay a
different conversation than the one that happened.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..media import DocumentPart, ImagePart, MediaPart
from ..models import (
    ContentPart,
    Message,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
)
from .errors import SessionFormatError

FORMAT_VERSION = 1
"""Bumped when the stored shape changes incompatibly. A reader refuses a payload with a
higher version rather than guessing at it."""

PartEncoder = Callable[[ContentPart], dict[str, Any]]
PartDecoder = Callable[[Mapping[str, Any]], ContentPart]


@dataclass(frozen=True)
class _Codec:
    cls: type[ContentPart]
    encode: PartEncoder
    decode: PartDecoder


_CODECS: dict[str, _Codec] = {}


def register_part_codec(
    part_type: type[ContentPart], encode: PartEncoder, decode: PartDecoder
) -> None:
    """Teach the session format a content part type.

    ``encode`` returns the part's fields as a JSON-serializable dict (the ``"type"`` key
    is added for it); ``decode`` rebuilds the part from that dict. Registering the same
    ``type`` twice replaces the earlier codec.
    """

    if not part_type.type:
        raise SessionFormatError(f"{part_type.__name__} has no 'type'; set a class-level type.")
    _CODECS[part_type.type] = _Codec(part_type, encode, decode)


register_part_codec(
    TextPart,
    lambda part: {"text": part.text},  # type: ignore[attr-defined]
    lambda data: TextPart(str(data["text"])),
)


def _encode_tool_call(part: ContentPart) -> dict[str, Any]:
    call = part.call  # type: ignore[attr-defined]
    return {
        "id": call.id,
        "name": call.name,
        "arguments": call.arguments,
        "raw_arguments": call.raw_arguments,
        "arguments_error": call.arguments_error,
    }


def _decode_tool_call(data: Mapping[str, Any]) -> ContentPart:
    return ToolCallPart(
        ToolCall(
            name=str(data["name"]),
            arguments=dict(data.get("arguments") or {}),
            id=str(data.get("id") or ""),
            raw_arguments=data.get("raw_arguments"),
            arguments_error=data.get("arguments_error"),
        )
    )


register_part_codec(ToolCallPart, _encode_tool_call, _decode_tool_call)
register_part_codec(
    ToolResultPart,
    lambda part: {
        "tool_call_id": part.tool_call_id,  # type: ignore[attr-defined]
        "content": part.content,  # type: ignore[attr-defined]
        "name": part.name,  # type: ignore[attr-defined]
        "is_error": part.is_error,  # type: ignore[attr-defined]
    },
    lambda data: ToolResultPart(
        tool_call_id=str(data["tool_call_id"]),
        content=str(data["content"]),
        name=str(data.get("name") or ""),
        is_error=bool(data.get("is_error", False)),
    ),
)

_MEDIA_EXTRAS: dict[type[MediaPart], str] = {ImagePart: "detail", DocumentPart: "filename"}


def _media_codec(cls: type[MediaPart]) -> tuple[PartEncoder, PartDecoder]:
    extra = _MEDIA_EXTRAS[cls]

    def encode(part: ContentPart) -> dict[str, Any]:
        assert isinstance(part, MediaPart)
        encoded: dict[str, Any] = {"media_type": part.media_type, extra: getattr(part, extra)}
        if part.data is not None:
            encoded["data"] = part.base64
        else:
            encoded["url"] = part.url
        return encoded

    def decode(data: Mapping[str, Any]) -> ContentPart:
        kwargs: dict[str, Any] = {"media_type": data.get("media_type") or None}
        if data.get(extra) is not None:
            kwargs[extra] = data[extra]
        if "data" in data:
            raw = base64.b64decode(str(data["data"]), validate=True)
            # A part built with a raised max_bytes was accepted once; reading it back
            # must not fail on the default limit.
            kwargs["max_bytes"] = max(cls.default_max_bytes, len(raw))
            return cls.from_bytes(raw, **kwargs)
        return cls.from_url(str(data["url"]), **kwargs)

    return encode, decode


register_part_codec(ImagePart, *_media_codec(ImagePart))
register_part_codec(DocumentPart, *_media_codec(DocumentPart))


def encode_message(message: Message) -> dict[str, Any]:
    encoded: dict[str, Any] = {"role": message.role, "metadata": dict(message.metadata)}
    if isinstance(message.content, str):
        encoded["content"] = message.content
    else:
        encoded["parts"] = [_encode_part(part) for part in message.parts]
    return encoded


def decode_message(data: Mapping[str, Any]) -> Message:
    try:
        role = data["role"]
        metadata = dict(data.get("metadata") or {})
        if "content" in data:
            return Message(role, str(data["content"]), metadata)
        parts = [_decode_part(part) for part in data["parts"]]
    except (KeyError, TypeError) as exc:
        raise SessionFormatError(f"Malformed stored message: {exc!r}.") from exc
    return Message(role, parts, metadata)


def encode_payload(
    messages: list[Message], state: dict[str, Any], metadata: dict[str, Any]
) -> str:
    """The session body as one JSON string. Raises :class:`SessionFormatError` when
    tool state, metadata or a message's metadata is not JSON-serializable -- at save
    time, rather than as an unreadable row later."""

    payload = {
        "format": FORMAT_VERSION,
        "messages": [encode_message(message) for message in messages],
        "state": state,
        "metadata": metadata,
    }
    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SessionFormatError(
            f"Session data must be JSON-serializable (tool state, metadata and message "
            f"metadata included): {exc}."
        ) from exc


def decode_payload(text: str) -> tuple[list[Message], dict[str, Any], dict[str, Any]]:
    """``(messages, state, metadata)`` from :func:`encode_payload`'s output."""

    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise SessionFormatError(f"Stored session is not valid JSON: {exc}.") from exc
    if not isinstance(payload, dict):
        raise SessionFormatError("Stored session is not a JSON object.")
    version = payload.get("format")
    if not isinstance(version, int) or version > FORMAT_VERSION or version < 1:
        raise SessionFormatError(
            f"Stored session has format version {version!r}; this kuhaku reads versions "
            f"up to {FORMAT_VERSION}. It was written by a newer kuhaku -- upgrade to read it."
        )
    messages = [decode_message(message) for message in payload.get("messages") or []]
    state = payload.get("state") or {}
    metadata = payload.get("metadata") or {}
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise SessionFormatError("Stored session state and metadata must be JSON objects.")
    return messages, dict(state), dict(metadata)


def _encode_part(part: ContentPart) -> dict[str, Any]:
    codec = _CODECS.get(part.type)
    if codec is None or not isinstance(part, codec.cls):
        raise SessionFormatError(
            f"No session codec is registered for content part type {part.type!r} "
            f"({type(part).__name__}); register one with register_part_codec()."
        )
    return {"type": part.type, **codec.encode(part)}


def _decode_part(data: Mapping[str, Any]) -> ContentPart:
    part_type = data.get("type")
    codec = _CODECS.get(part_type) if isinstance(part_type, str) else None
    if codec is None:
        raise SessionFormatError(
            f"Stored session contains a content part of unknown type {part_type!r}; "
            f"known types: {', '.join(sorted(_CODECS))}. It was written by code that "
            f"registered a codec for it -- register the same codec to read it."
        )
    try:
        return codec.decode({key: value for key, value in data.items() if key != "type"})
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionFormatError(
            f"Stored {part_type!r} content part is malformed: {exc}."
        ) from exc
