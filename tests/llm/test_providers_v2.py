"""Wire-level tests of the v2 methods on the HTTP providers (OpenAI, Groq, Anthropic,
Ollama): request payloads, native structured-output modes, streaming parsers, and
that retry / circuit breaker / usage apply to the new paths too.

No real network calls -- ``requests.post`` is monkeypatched.
"""

from __future__ import annotations

import json

import pytest
import requests

from kuhaku.core.llm import (
    GenerationOptions,
    LLMError,
    StructuredOutputMode,
    TokenUsage,
    UnsupportedContentError,
)
from kuhaku.core.llm.anthropic_provider import AnthropicProvider
from kuhaku.core.llm.groq_provider import GroqProvider
from kuhaku.core.llm.ollama_provider import OllamaProvider
from kuhaku.core.llm.openai_provider import OpenAIProvider
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.models import ContentPart, Message

_FAST = {"retry_backoff_base_seconds": 0.001, "retry_backoff_max_seconds": 0.01}
_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


class FakeResponse:
    def __init__(self, payload=None, lines=None):
        self._payload = payload
        self._lines = lines or []
        self.closed = False

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True


class Recorder:
    """A ``requests.post`` stand-in that records calls and replays responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def payload(self) -> dict:
        return self.calls[-1]["json"]


def _sse(*events) -> list[bytes]:
    lines: list[bytes] = []
    for event in events:
        data = event if isinstance(event, str) else json.dumps(event)
        lines += [f"data: {data}".encode(), b""]
    return lines


def _openai(content: str, usage=None, finish="stop"):
    payload = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    if usage:
        payload["usage"] = usage
    return FakeResponse(payload)


class _ImagePart(ContentPart):
    type = "image"


# --- generate() payloads are unchanged ------------------------------------------------


def test_openai_generate_payload_is_unchanged(monkeypatch):
    post = Recorder(_openai("hi"))
    monkeypatch.setattr(requests, "post", post)
    OpenAIProvider(api_key="k", model="m").generate("sys", "usr")
    assert post.payload == {
        "model": "m",
        "temperature": 0.1,
        "max_tokens": 1024,
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
    }
    assert "stream" not in post.calls[-1]


def test_anthropic_generate_payload_is_unchanged(monkeypatch):
    post = Recorder(FakeResponse({"content": [{"type": "text", "text": "hi"}]}))
    monkeypatch.setattr(requests, "post", post)
    AnthropicProvider(api_key="k", model="m").generate("sys", "usr")
    # No temperature: the provider sends one only when it is configured (and the model
    # accepts it) -- current Claude models reject a non-default temperature with a 400.
    assert post.payload == {
        "model": "m",
        "max_tokens": 1024,
        "system": "sys",
        "messages": [{"role": "user", "content": "usr"}],
    }


def test_ollama_generate_payload_is_unchanged(monkeypatch):
    post = Recorder(FakeResponse({"message": {"content": "hi"}}))
    monkeypatch.setattr(requests, "post", post)
    OllamaProvider("http://x", "m").generate("sys", "usr")
    assert post.payload == {
        "model": "m",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }


# --- capabilities --------------------------------------------------------------------


def test_capabilities_per_provider():
    assert OpenAIProvider(api_key="k", model="m").capabilities.structured_output is (
        StructuredOutputMode.SCHEMA
    )
    assert GroqProvider(api_key="k", model="m").capabilities.structured_output is (
        StructuredOutputMode.JSON
    )
    assert AnthropicProvider(api_key="k", model="m").capabilities.structured_output is (
        StructuredOutputMode.SCHEMA
    )
    ollama = OllamaProvider("http://x", "m").capabilities
    assert ollama.structured_output is StructuredOutputMode.SCHEMA
    assert ollama.chat
    # tool_calling became native on every built-in provider with feat/tool-calling.
    assert ollama.content_types == frozenset({"text"}) and ollama.tool_calling


def test_openai_native_structured_output_can_be_turned_off(monkeypatch):
    provider = OpenAIProvider(api_key="k", model="m", native_structured_output=False)
    assert provider.capabilities.structured_output is StructuredOutputMode.NONE
    assert OpenAIProvider.capabilities.structured_output is StructuredOutputMode.SCHEMA
    post = Recorder(_openai('{"name": "a"}'))
    monkeypatch.setattr(requests, "post", post)
    result = provider.generate_structured([Message.user("q")], _SCHEMA)
    assert result.native is False and result.data == {"name": "a"}
    assert "response_format" not in post.payload


# --- chat ----------------------------------------------------------------------------


def test_openai_chat_sends_turns_and_options(monkeypatch):
    post = Recorder(_openai(" ok ", usage={"prompt_tokens": 5, "completion_tokens": 1}))
    monkeypatch.setattr(requests, "post", post)
    response = OpenAIProvider(api_key="k", model="m").chat(
        [Message.user("a"), Message.assistant("b"), Message.user("c")],
        options=GenerationOptions(temperature=0.7, max_tokens=9),
    )
    assert response.text == "ok"
    assert response.usage == TokenUsage(5, 1)
    assert response.finish_reason == "stop"
    assert response.model == "openai:m"
    assert post.payload["messages"] == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert post.payload["temperature"] == 0.7 and post.payload["max_tokens"] == 9


def test_anthropic_chat_lifts_system_messages_and_omits_absent_system(monkeypatch):
    post = Recorder(
        FakeResponse({"content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn"}),
        FakeResponse({"content": [{"type": "text", "text": "y"}]}),
    )
    monkeypatch.setattr(requests, "post", post)
    provider = AnthropicProvider(api_key="k", model="m")
    response = provider.chat([Message.system("one"), Message.system("two"), Message.user("q")])
    assert response.finish_reason == "end_turn"
    assert post.calls[0]["json"]["system"] == "one\n\ntwo"
    provider.chat([Message.user("q")])
    assert "system" not in post.payload


def test_providers_reject_unsupported_content_before_any_request(monkeypatch):
    post = Recorder()
    monkeypatch.setattr(requests, "post", post)
    for provider in (
        OpenAIProvider(api_key="k", model="m"),
        AnthropicProvider(api_key="k", model="m"),
        OllamaProvider("http://x", "m"),
    ):
        with pytest.raises(UnsupportedContentError):
            provider.chat([Message.user([_ImagePart()])])
        with pytest.raises(UnsupportedContentError):
            provider.chat([Message("tool", "result")])
    assert post.calls == []


# --- native structured output -----------------------------------------------------------


def test_openai_structured_uses_json_schema_response_format(monkeypatch):
    post = Recorder(_openai('{"name": "Ada"}', usage={"prompt_tokens": 2, "completion_tokens": 3}))
    monkeypatch.setattr(requests, "post", post)
    result = OpenAIProvider(api_key="k", model="m").generate_structured(
        [Message.user("who?")], _SCHEMA, name="person"
    )
    assert result.data == {"name": "Ada"}
    assert result.native and result.attempts == 1
    assert result.usage == TokenUsage(2, 3)
    assert post.payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "person", "schema": _SCHEMA, "strict": False},
    }
    assert post.payload["messages"] == [{"role": "user", "content": "who?"}]


def test_groq_structured_uses_json_mode_with_schema_in_prompt(monkeypatch):
    post = Recorder(_openai('{"name": "Ada"}'))
    monkeypatch.setattr(requests, "post", post)
    result = GroqProvider(api_key="k", model="m").generate_structured(
        [Message.user("who?")], _SCHEMA
    )
    assert result.data == {"name": "Ada"}
    assert post.payload["response_format"] == {"type": "json_object"}
    system = post.payload["messages"][0]
    assert system["role"] == "system" and "JSON" in system["content"]
    assert '"required"' in system["content"]


def test_anthropic_structured_forces_a_tool_call(monkeypatch):
    reply = {
        "content": [{"type": "tool_use", "name": "person", "input": {"name": "Ada"}}],
        "usage": {"input_tokens": 9, "output_tokens": 4},
    }
    post = Recorder(FakeResponse(reply))
    monkeypatch.setattr(requests, "post", post)
    result = AnthropicProvider(api_key="k", model="m").generate_structured(
        [Message.user("who?")], _SCHEMA, name="person"
    )
    assert result.data == {"name": "Ada"}
    assert result.usage == TokenUsage(9, 4)
    assert post.payload["tools"][0]["name"] == "person"
    assert post.payload["tools"][0]["input_schema"] == _SCHEMA
    assert post.payload["tool_choice"] == {"type": "tool", "name": "person"}


def test_anthropic_structured_wraps_a_non_object_schema(monkeypatch):
    schema = {"type": "array", "items": {"type": "integer"}}
    reply = {"content": [{"type": "tool_use", "name": "response", "input": {"value": [1, 2]}}]}
    post = Recorder(FakeResponse(reply))
    monkeypatch.setattr(requests, "post", post)
    result = AnthropicProvider(api_key="k", model="m").generate_structured(
        [Message.user("q")], schema
    )
    assert result.data == [1, 2]
    assert post.payload["tools"][0]["input_schema"]["properties"]["value"] == schema


def test_ollama_structured_passes_schema_as_format(monkeypatch):
    post = Recorder(FakeResponse({"message": {"content": '{"name": "Ada"}'}}))
    monkeypatch.setattr(requests, "post", post)
    result = OllamaProvider("http://x", "m").generate_structured([Message.user("q")], _SCHEMA)
    assert result.data == {"name": "Ada"}
    assert post.payload["format"] == _SCHEMA


def test_native_structured_repairs_unparseable_output(monkeypatch):
    post = Recorder(_openai('{"name": "Ad'), _openai('{"name": "Ada"}'))
    monkeypatch.setattr(requests, "post", post)
    result = OpenAIProvider(api_key="k", model="m").generate_structured(
        [Message.user("q")], _SCHEMA
    )
    assert result.data == {"name": "Ada"} and result.attempts == 2
    assert post.payload["messages"][-2] == {"role": "assistant", "content": '{"name": "Ad'}


def test_structured_call_is_retried_like_generate(monkeypatch):
    post = Recorder(requests.ConnectionError("blip"), _openai('{"name": "Ada"}'))
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(api_key="k", model="m", retry_max_attempts=3, **_FAST)
    assert provider.generate_structured([Message.user("q")], _SCHEMA).data == {"name": "Ada"}
    assert len(post.calls) == 2


def test_structured_call_goes_through_the_circuit_breaker(monkeypatch):
    post = Recorder(requests.ConnectionError("down"))
    monkeypatch.setattr(requests, "post", post)
    provider = OllamaProvider(
        "http://x", "m", retry_enabled=False, circuit_breaker_failure_threshold=1
    )
    with pytest.raises(LLMError):
        provider.generate_structured([Message.user("q")], _SCHEMA)
    with pytest.raises(LLMError, match="(?i)circuit"):
        provider.generate_structured([Message.user("q")], _SCHEMA)
    assert len(post.calls) == 1


# --- streaming -------------------------------------------------------------------------


# --- async and token tracking over real providers ---------------------------------------


def test_token_tracking_wraps_a_real_provider_v2(monkeypatch):
    post = Recorder(_openai('{"name": "Ada"}', usage={"prompt_tokens": 1, "completion_tokens": 1}))
    monkeypatch.setattr(requests, "post", post)
    tracker = TokenTrackingLLM(OpenAIProvider(api_key="k", model="m"), provider="openai")
    assert tracker.capabilities.structured_output is StructuredOutputMode.SCHEMA
    result = tracker.generate_structured([Message.user("q")], _SCHEMA)
    assert result.native and result.usage == TokenUsage(1, 1)
