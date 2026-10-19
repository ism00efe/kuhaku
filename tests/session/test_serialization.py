"""The versioned stored form of messages, and the part-codec seam."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

import pytest

from kuhaku.core.models import ContentPart, Message, TextPart
from kuhaku.core.session import (
    FORMAT_VERSION,
    SessionFormatError,
    register_part_codec,
    serialization,
)
from kuhaku.core.session.serialization import (
    decode_message,
    decode_payload,
    encode_message,
    encode_payload,
)


@dataclass(frozen=True)
class FakeImagePart(ContentPart):
    url: str

    type: ClassVar[str] = "fake_image"


@pytest.fixture
def isolated_codecs(monkeypatch):
    monkeypatch.setattr(serialization, "_CODECS", dict(serialization._CODECS))


@pytest.mark.parametrize(
    "message",
    [
        Message.user("plain string"),
        Message("assistant", [TextPart("a"), TextPart("b")], {"k": [1, 2]}),
        Message("tool", "custom role survives"),
        Message.user(""),
    ],
)
def test_messages_round_trip_exactly(message):
    assert decode_message(encode_message(message)) == message


def test_payload_carries_its_format_version():
    payload = json.loads(encode_payload([Message.user("x")], {"t": {"a": 1}}, {"m": True}))
    assert payload["format"] == FORMAT_VERSION
    assert decode_payload(json.dumps(payload)) == ([Message.user("x")], {"t": {"a": 1}}, {"m": True})


@pytest.mark.parametrize("version", [FORMAT_VERSION + 1, 0, "1", None])
def test_unknown_format_versions_are_refused(version):
    with pytest.raises(SessionFormatError, match="format version"):
        decode_payload(json.dumps({"format": version, "messages": []}))


def test_an_unknown_stored_part_type_is_an_error_naming_it():
    stored = {"format": 1, "messages": [{"role": "user", "parts": [{"type": "hologram", "url": "x"}]}]}
    with pytest.raises(SessionFormatError, match="'hologram'"):
        decode_payload(json.dumps(stored))


def test_encoding_a_part_without_a_codec_is_an_error_not_a_silent_drop():
    with pytest.raises(SessionFormatError, match="fake_image"):
        encode_message(Message.user([FakeImagePart("http://x")]))


def test_a_registered_codec_makes_a_new_part_type_round_trip(isolated_codecs):
    register_part_codec(
        FakeImagePart,
        lambda part: {"url": part.url},  # type: ignore[attr-defined]
        lambda data: FakeImagePart(data["url"]),
    )
    message = Message.user([TextPart("look:"), FakeImagePart("http://x/cat.png")])
    encoded = encode_message(message)
    assert encoded["parts"][1] == {"type": "fake_image", "url": "http://x/cat.png"}
    assert decode_message(json.loads(json.dumps(encoded))) == message


def test_malformed_messages_and_json_are_format_errors():
    with pytest.raises(SessionFormatError):
        decode_message({"content": "no role"})
    with pytest.raises(SessionFormatError):
        decode_payload("{not json")
    with pytest.raises(SessionFormatError):
        decode_payload("[]")


def test_non_serializable_data_is_refused_at_encode_time():
    with pytest.raises(SessionFormatError):
        encode_payload([], {}, {"when": object()})
    with pytest.raises(SessionFormatError):
        encode_payload([], {"t": {"x": float("nan")}}, {})


def test_tool_and_media_parts_round_trip():
    # Session memory x tool calling x multimodal: a stored tool conversation (and its
    # attachments) replays exactly.
    from kuhaku import DocumentPart, ImagePart
    from kuhaku.core.models import ToolCall, ToolCallPart, ToolResultPart
    from tests.test_media import PDF, tiny_png

    flagged = ToolCall("f", {}, id="c2", raw_arguments="not json", arguments_error="bad")
    messages = [
        Message.user(
            [
                ImagePart.from_bytes(tiny_png(), detail="low"),
                ImagePart.from_url("https://example.com/a.png"),
                DocumentPart.from_bytes(PDF, filename="invoice.pdf"),
                TextPart("q"),
            ]
        ),
        Message.assistant(
            [TextPart("x"), ToolCallPart(ToolCall("get", {"a": 1}, id="c1")), ToolCallPart(flagged)]
        ),
        Message("tool", [ToolResultPart("c1", "out", name="get", is_error=True)]),
    ]
    text = encode_payload(messages, {}, {})
    assert "invoice.pdf" in text
    decoded, _, _ = decode_payload(text)
    assert decoded == messages


def test_a_malformed_stored_media_part_is_a_format_error():
    stored = {"format": 1, "messages": [{"role": "user", "parts": [{"type": "image", "url": "x"}]}]}
    with pytest.raises(SessionFormatError, match="'image'"):
        decode_payload(json.dumps(stored))


def test_sanitize_message_masks_tool_results():
    from kuhaku.core.models import ToolResultPart
    from kuhaku.core.session import sanitize_message

    message = Message("tool", [ToolResultPart("c1", "mail jane.doe@example.com now")])
    cleaned = sanitize_message(message)
    assert "jane.doe@example.com" not in cleaned.parts[0].content
    assert cleaned.parts[0].tool_call_id == "c1"
