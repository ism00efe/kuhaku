"""Images and documents on the tool-calling path: a user turn may carry media beside
tool calls and results, encoded exactly as ``chat`` encodes it, and the same rules
apply (declared content types only, user turns only). No network."""

from __future__ import annotations

import json

import pytest
import requests

from kuhaku import DocumentPart, ImagePart, Message, TextPart
from kuhaku.core.llm import LegacyLLMAdapter, ToolSpec, UnsupportedContentError
from kuhaku.core.llm.anthropic_provider import AnthropicProvider
from kuhaku.core.llm.ollama_provider import OllamaProvider
from kuhaku.core.llm.openai_provider import OpenAIProvider
from kuhaku.core.llm.tool_calling import split_tool_message
from kuhaku.core.models import ToolCall, ToolCallPart
from tests.test_media import tiny_png

from .test_multimodal import ANTHROPIC_REPLY, OLLAMA_REPLY, OPENAI_REPLY, Recorder

WEATHER = ToolSpec(
    "get_weather",
    "Current weather.",
    {"type": "object", "properties": {"city": {"type": "string"}}},
)
CALL = ToolCall("get_weather", {"city": "Oslo"}, id="c1")


@pytest.fixture
def image() -> ImagePart:
    return ImagePart.from_bytes(tiny_png())


def _history(image: ImagePart) -> list[Message]:
    return [
        Message.user([image, TextPart("Where is this? Then get the weather.")]),
        Message.assistant([TextPart("checking"), ToolCallPart(CALL)]),
        Message.tool_result("c1", "12C"),
    ]


def test_split_tool_message_flags_media_on_user_turns_only(image):
    split = split_tool_message(Message.user([image, TextPart("q")]), provider="p")
    assert split.has_media and split.text == "q"
    assert not split_tool_message(Message.user("q"), provider="p").has_media
    with pytest.raises(UnsupportedContentError, match="assistant"):
        split_tool_message(Message.assistant([image]), provider="p")


def test_openai_tool_payload_carries_user_images(monkeypatch, image):
    post = Recorder(OPENAI_REPLY)
    monkeypatch.setattr(requests, "post", post)
    OpenAIProvider(api_key="k", model="gpt-4o").chat_with_tools(_history(image), [WEATHER])
    messages = post.payload["messages"]
    assert messages[0] == {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image.data_url}},
            {"type": "text", "text": "Where is this? Then get the weather."},
        ],
    }
    assert messages[2]["role"] == "tool"
    assert "tools" in post.payload


def test_anthropic_tool_payload_carries_user_images(monkeypatch, image):
    post = Recorder(ANTHROPIC_REPLY)
    monkeypatch.setattr(requests, "post", post)
    AnthropicProvider(api_key="k", model="claude-sonnet-5").chat_with_tools(
        _history(image), [WEATHER]
    )
    first = post.payload["messages"][0]
    assert first["role"] == "user"
    assert first["content"][0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": image.base64},
    }
    assert post.payload["messages"][2]["content"][0]["type"] == "tool_result"


def test_ollama_tool_payload_carries_user_images(monkeypatch, image):
    post = Recorder(OLLAMA_REPLY)
    monkeypatch.setattr(requests, "post", post)
    OllamaProvider("http://x", "llava").chat_with_tools(_history(image), [WEATHER])
    assert post.payload["messages"][0] == {
        "role": "user",
        "content": "Where is this? Then get the weather.",
        "images": [image.base64],
    }


def test_undeclared_media_is_still_rejected_on_the_tool_path(monkeypatch, image):
    post = Recorder(OLLAMA_REPLY)
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(UnsupportedContentError):
        OllamaProvider("http://x", "llama3").chat_with_tools(_history(image), [WEATHER])
    pdf = DocumentPart.from_bytes(b"%PDF-1.7 x", filename="a.pdf")
    with pytest.raises(UnsupportedContentError):
        OllamaProvider("http://x", "llava").chat_with_tools(
            [Message.user([pdf, TextPart("q")])], [WEATHER]
        )
    assert post.calls == []


def test_vertex_tool_contents_carry_user_media(monkeypatch, image):
    pytest.importorskip("google.genai", reason="needs the optional 'vertex' extra")
    from google import genai

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    from .test_vertex_tool_calling import _reply
    from .test_vertex_v2 import Models

    models = Models(reply=_reply("ok"))
    monkeypatch.setattr(genai, "Client", lambda **_: type("C", (), {"models": models})())
    VertexAIProvider(project="p", location="l").chat_with_tools(_history(image), [WEATHER])
    assert models.calls[0]["contents"][0] == {
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": "image/png", "data": image.data}},
            {"text": "Where is this? Then get the weather."},
        ],
    }


def test_emulated_tool_calling_rejects_media_a_text_only_provider_cannot_send(image):
    class Model:
        name = "scripted:v1"

        def generate(self, system: str, user: str) -> str:
            return json.dumps({"answer": "fine"})

    llm = LegacyLLMAdapter(Model())
    with pytest.raises(UnsupportedContentError):
        llm.chat_with_tools([Message.user([image, TextPart("q")])], [WEATHER])
