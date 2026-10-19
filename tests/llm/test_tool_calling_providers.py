"""Wire-level tests of native tool calling on the HTTP providers (OpenAI, Groq,
Anthropic, Ollama): request encoding, reply parsing, streamed tool calls, and that
retry / circuit breaker apply. ``requests.post`` is monkeypatched -- no network."""

from __future__ import annotations

import pytest
import requests

from kuhaku.core.llm import LLMError, ToolSpec, UnsupportedContentError
from kuhaku.core.llm.anthropic_provider import AnthropicProvider
from kuhaku.core.llm.groq_provider import GroqProvider
from kuhaku.core.llm.ollama_provider import OllamaProvider
from kuhaku.core.llm.openai_provider import OpenAIProvider
from kuhaku.core.models import Message, TextPart, ToolCall, ToolCallPart

from .test_providers_v2 import _FAST, FakeResponse, Recorder, _openai

PARAMS = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
WEATHER = ToolSpec("get_weather", "Current weather.", PARAMS)
TIME = ToolSpec("get_time", "Current time.")
CALL = ToolCall("get_weather", {"city": "Oslo"}, id="c1")

# A user question, the model's tool call, and the call's result.
HISTORY = [
    Message.system("sys"),
    Message.user("weather?"),
    Message.assistant([TextPart("checking"), ToolCallPart(CALL)]),
    Message.tool_result("c1", "12C"),
]


def _openai_calls(*calls, content=None):
    return FakeResponse(
        {
            "choices": [
                {
                    "message": {"content": content, "tool_calls": list(calls)},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        }
    )


def _fn(id, name, arguments):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


# --- OpenAI / Groq --------------------------------------------------------------------


def test_openai_tool_payload_encodes_tools_choice_and_history(monkeypatch):
    post = Recorder(_openai_calls(_fn("c2", "get_time", "{}")))
    monkeypatch.setattr(requests, "post", post)
    response = OpenAIProvider(api_key="k", model="m").chat_with_tools(HISTORY, [WEATHER, TIME])
    payload = post.payload
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "Current weather.", "parameters": PARAMS},
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Current time.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    assert payload["tool_choice"] == "auto"
    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [_fn("c1", "get_weather", '{"city": "Oslo"}')],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "12C"},
    ]
    assert response.text == "" and response.finish_reason == "tool_calls"
    assert response.tool_calls == (ToolCall("get_time", {}, id="c2"),)
    assert response.metadata == {"tool_calling": "native"}
    assert response.usage.prompt_tokens == 9


def test_openai_tool_choice_variants(monkeypatch):
    post = Recorder(_openai("a"), _openai("b"), _openai("c"))
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(api_key="k", model="m")
    provider.chat_with_tools([Message.user("q")], [WEATHER], tool_choice="get_weather")
    assert post.payload["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    provider.chat_with_tools([Message.user("q")], [WEATHER], tool_choice="required")
    assert post.payload["tool_choice"] == "required"
    response = provider.chat_with_tools([Message.user("q")], [WEATHER], tool_choice="none")
    assert post.payload["tool_choice"] == "none" and response.text == "c"
    assert response.tool_calls == ()


def test_openai_malformed_arguments_are_flagged_not_dropped(monkeypatch):
    post = Recorder(
        _openai_calls(_fn("c1", "get_weather", '{"city": '), _fn("c2", "get_time", "")),
        _openai("done"),
    )
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(api_key="k", model="m")
    response = provider.chat_with_tools([Message.user("q")], [WEATHER, TIME])
    bad, good = response.tool_calls
    assert not bad.valid and bad.raw_arguments == '{"city": ' and bad.id == "c1"
    assert good.valid and good.arguments == {}
    # Sent back as the model wrote it.
    provider.chat_with_tools([Message.user("q"), response.message], [WEATHER, TIME])
    sent = post.payload["messages"][1]
    assert sent["content"] is None
    assert sent["tool_calls"][0]["function"]["arguments"] == '{"city": '


def test_plain_chat_still_rejects_tool_turns_and_payload_has_no_tools(monkeypatch):
    post = Recorder(_openai("hi"))
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(api_key="k", model="m")
    with pytest.raises(UnsupportedContentError):
        provider.chat(HISTORY)
    provider.chat([Message.user("q")])
    assert "tools" not in post.payload and "tool_choice" not in post.payload
    assert post.payload["messages"] == [{"role": "user", "content": "q"}]


def test_openai_rejects_two_results_in_one_tool_message(monkeypatch):
    from kuhaku.core.models import ToolResultPart

    monkeypatch.setattr(requests, "post", Recorder())
    message = Message("tool", [ToolResultPart("a", "1"), ToolResultPart("b", "2")])
    with pytest.raises(UnsupportedContentError, match="one tool result"):
        OpenAIProvider(api_key="k", model="m").chat_with_tools([message], [WEATHER])


def test_groq_inherits_native_tool_calling(monkeypatch):
    post = Recorder(_openai_calls(_fn("c1", "get_weather", '{"city": "x"}')))
    monkeypatch.setattr(requests, "post", post)
    provider = GroqProvider(api_key="k", model="m")
    assert provider.capabilities.tool_calling
    response = provider.chat_with_tools([Message.user("q")], [WEATHER], tool_choice="required")
    assert response.tool_calls[0].arguments == {"city": "x"}
    assert post.payload["tool_choice"] == "required"
    assert post.calls[-1]["url"].startswith("https://api.groq.com")


def test_openai_native_tool_calling_can_be_turned_off(monkeypatch):
    post = Recorder(_openai('{"tool_calls": [{"name": "get_weather", "arguments": {"city": "x"}}]}'))
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(api_key="k", model="m", native_tool_calling=False)
    assert not provider.capabilities.tool_calling
    assert provider.capabilities.structured_output.value == "schema"
    response = provider.chat_with_tools([Message.user("q")], [WEATHER])
    assert response.metadata == {"tool_calling": "emulated"}
    assert response.tool_calls[0].arguments == {"city": "x"}
    assert "tools" not in post.payload
    assert post.payload["response_format"]["type"] == "json_schema"


def test_tool_call_is_retried_and_goes_through_the_circuit_breaker(monkeypatch):
    post = Recorder(requests.ConnectionError("blip"), _openai_calls(_fn("c", "get_time", "{}")))
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(api_key="k", model="m", retry_max_attempts=3, **_FAST)
    assert provider.chat_with_tools([Message.user("q")], [TIME]).tool_calls
    assert len(post.calls) == 2

    post = Recorder(requests.ConnectionError("down"))
    monkeypatch.setattr(requests, "post", post)
    provider = OllamaProvider("http://x", "m", retry_enabled=False, circuit_breaker_failure_threshold=1)
    with pytest.raises(LLMError):
        provider.chat_with_tools([Message.user("q")], [TIME])
    with pytest.raises(LLMError, match="(?i)circuit"):
        provider.chat_with_tools([Message.user("q")], [TIME])
    assert len(post.calls) == 1


# --- Anthropic ------------------------------------------------------------------------


def test_anthropic_tool_payload_and_reply(monkeypatch):
    reply = {
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Oslo"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    post = Recorder(FakeResponse(reply))
    monkeypatch.setattr(requests, "post", post)
    history = [*HISTORY, Message.tool_result("c9", "fail", name="get_time", is_error=True)]
    response = AnthropicProvider(api_key="k", model="m").chat_with_tools(history, [WEATHER])
    payload = post.payload
    assert payload["system"] == "sys"
    assert payload["tools"] == [
        {"name": "get_weather", "description": "Current weather.", "input_schema": PARAMS}
    ]
    assert payload["tool_choice"] == {"type": "auto"}
    assert payload["messages"] == [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "Oslo"}},
            ],
        },
        # Consecutive tool results merged into one user turn.
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "12C"},
                {"type": "tool_result", "tool_use_id": "c9", "content": "fail", "is_error": True},
            ],
        },
    ]
    assert response.text == "Checking." and response.finish_reason == "tool_use"
    assert response.tool_calls == (ToolCall("get_weather", {"city": "Oslo"}, id="tu_1"),)


def test_anthropic_tool_choice_mapping(monkeypatch):
    text = {"content": [{"type": "text", "text": "x"}]}
    post = Recorder(*(FakeResponse(text) for _ in range(3)))
    monkeypatch.setattr(requests, "post", post)
    provider = AnthropicProvider(api_key="k", model="m")
    expected = {
        "required": {"type": "any"},
        "none": {"type": "none"},
        "get_weather": {"type": "tool", "name": "get_weather"},
    }
    for choice, wire in expected.items():
        provider.chat_with_tools([Message.user("q")], [WEATHER], tool_choice=choice)
        assert post.payload["tool_choice"] == wire


def test_anthropic_structured_output_does_not_surface_its_tool_as_a_call(monkeypatch):
    reply = {"content": [{"type": "tool_use", "id": "t", "name": "response", "input": {"name": "A"}}]}
    monkeypatch.setattr(requests, "post", Recorder(FakeResponse(reply)))
    result = AnthropicProvider(api_key="k", model="m").generate_structured(
        [Message.user("q")], {"type": "object", "properties": {"name": {"type": "string"}}}
    )
    assert result.data == {"name": "A"}


# --- Ollama ---------------------------------------------------------------------------


def _ollama_reply(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return FakeResponse({"message": message, "done_reason": "stop"})


def test_ollama_tool_payload_and_reply(monkeypatch):
    post = Recorder(
        _ollama_reply(tool_calls=[{"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}])
    )
    monkeypatch.setattr(requests, "post", post)
    response = OllamaProvider("http://x", "m").chat_with_tools(HISTORY, [WEATHER])
    payload = post.payload
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "Current weather.", "parameters": PARAMS},
        }
    ]
    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}],
        },
        {"role": "tool", "content": "12C", "tool_name": "get_weather"},
    ]
    (call,) = response.tool_calls
    assert call.arguments == {"city": "Oslo"} and call.id.startswith("call_")
    assert response.metadata == {"tool_calling": "native"}


def test_ollama_tool_choice_is_approximated(monkeypatch):
    post = Recorder(_ollama_reply("a"), _ollama_reply("b"), _ollama_reply("c"))
    monkeypatch.setattr(requests, "post", post)
    provider = OllamaProvider("http://x", "m")

    response = provider.chat_with_tools([Message.user("q")], [WEATHER, TIME], tool_choice="none")
    assert "tools" not in post.payload and response.metadata == {"tool_calling": "native"}

    response = provider.chat_with_tools(
        [Message.system("s"), Message.user("q")], [WEATHER, TIME], tool_choice="get_time"
    )
    assert [t["function"]["name"] for t in post.payload["tools"]] == ["get_time"]
    assert post.payload["messages"][1] == {"role": "system", "content": 'Call the tool "get_time" now.'}
    assert response.metadata["tool_choice_enforced"] is False

    provider.chat_with_tools([Message.user("q")], [WEATHER, TIME], tool_choice="required")
    assert len(post.payload["tools"]) == 2
    assert post.payload["messages"][0]["role"] == "system"


def test_ollama_string_arguments_are_parsed_or_flagged(monkeypatch):
    post = Recorder(
        _ollama_reply(tool_calls=[
            {"function": {"name": "get_weather", "arguments": '{"city": "y"}'}},
            {"function": {"name": "get_weather", "arguments": "nope"}},
        ])
    )
    monkeypatch.setattr(requests, "post", post)
    good, bad = OllamaProvider("http://x", "m").chat_with_tools(
        [Message.user("q")], [WEATHER]
    ).tool_calls
    assert good.arguments == {"city": "y"} and not bad.valid
