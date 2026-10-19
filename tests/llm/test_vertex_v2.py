"""v2 methods on the Vertex AI provider: conversation contents, native structured
output, streaming. ``google.genai.Client`` is monkeypatched -- no network."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "google.genai",
    reason="Vertex AI provider tests need the optional 'vertex' extra.",
)

from kuhaku.core.llm import (
    GenerationOptions,
    StructuredOutputMode,
    TokenUsage,
)
from kuhaku.core.models import Message


class Usage:
    def __init__(self, prompt, completion):
        self.prompt_token_count = prompt
        self.candidates_token_count = completion


class Reply:
    def __init__(self, text, usage=None):
        self.text = text
        self.usage_metadata = usage
        self.candidates = []


class Models:
    def __init__(self, reply=None, chunks=None, stream_error=None):
        self.reply = reply
        self.chunks = chunks or []
        self.stream_error = stream_error
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply

    def generate_content_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if self.stream_error is not None:
            error, self.stream_error = self.stream_error, None
            raise error
        yield from self.chunks


@pytest.fixture
def make_provider(monkeypatch):
    from google import genai

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    def make(models, **kwargs):
        monkeypatch.setattr(genai, "Client", lambda **_: type("C", (), {"models": models})())
        return VertexAIProvider(project="p", location="l", **kwargs)

    return make


def test_capabilities(make_provider):
    caps = make_provider(Models()).capabilities
    assert caps.structured_output is StructuredOutputMode.SCHEMA
    assert caps.chat


def test_chat_sends_a_conversation_as_contents_with_options(make_provider):
    models = Models(reply=Reply(" ok ", Usage(3, 1)))
    response = make_provider(models).chat(
        [Message.system("s"), Message.user("a"), Message.assistant("b"), Message.user("c")],
        options=GenerationOptions(temperature=0.2, max_tokens=50),
    )
    assert response.text == "ok"
    assert response.usage == TokenUsage(3, 1)
    call = models.calls[0]
    assert call["contents"] == [
        {"role": "user", "parts": [{"text": "a"}]},
        {"role": "model", "parts": [{"text": "b"}]},
        {"role": "user", "parts": [{"text": "c"}]},
    ]
    assert call["config"]["system_instruction"] == "s"
    assert call["config"]["temperature"] == 0.2
    assert call["config"]["max_output_tokens"] == 50


def test_structured_sets_json_mime_type_and_schema(make_provider):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    models = Models(reply=Reply('{"n": 7}'))
    result = make_provider(models).generate_structured([Message.user("q")], schema)
    assert result.data == {"n": 7} and result.native
    config = models.calls[0]["config"]
    assert config["response_mime_type"] == "application/json"
    assert config.get("response_json_schema", config.get("response_schema")) == schema


