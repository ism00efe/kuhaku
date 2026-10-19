"""Native tool calling on the Vertex AI provider. ``google.genai.Client`` is
monkeypatched -- no network."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "google.genai",
    reason="Vertex AI provider tests need the optional 'vertex' extra.",
)

from kuhaku.core.llm import LLMError, ToolSpec, UnsupportedContentError
from kuhaku.core.models import Message, TextPart, ToolCall, ToolCallPart

from .test_vertex_v2 import Models, Reply, Usage


@pytest.fixture
def make_provider(monkeypatch):
    from google import genai

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    def make(models, **kwargs):
        monkeypatch.setattr(genai, "Client", lambda **_: type("C", (), {"models": models})())
        return VertexAIProvider(project="p", location="l", **kwargs)

    return make


PARAMS = {"type": "object", "properties": {"city": {"type": "string"}}}
WEATHER = ToolSpec("get_weather", "Current weather.", PARAMS)
TIME = ToolSpec("get_time")


class FunctionCall:
    def __init__(self, name, args, id=None):
        self.name = name
        self.args = args
        self.id = id


def _reply(text=None, calls=(), usage=None):
    reply = Reply(text, usage)
    reply.function_calls = list(calls) or None
    return reply


def test_capabilities(make_provider):
    assert make_provider(Models()).capabilities.tool_calling


def test_tool_request_config_and_contents(make_provider):
    models = Models(reply=_reply(calls=[FunctionCall("get_weather", {"city": "Oslo"})], usage=Usage(4, 2)))
    call = ToolCall("get_weather", {"city": "Oslo"}, id="c1")
    history = [
        Message.system("s"),
        Message.user("weather?"),
        Message.assistant([TextPart("checking"), ToolCallPart(call)]),
        Message.tool_result("c1", "12C"),
        Message.tool_result("c2", "no clock", name="get_time", is_error=True),
    ]
    response = make_provider(models).chat_with_tools(history, [WEATHER, TIME])
    request = models.calls[0]
    config = request["config"]
    declaration = config["tools"][0]["function_declarations"][0]
    assert declaration["name"] == "get_weather" and declaration["description"] == "Current weather."
    assert declaration.get("parameters_json_schema", declaration.get("parameters")) == PARAMS
    assert config["tool_config"] == {"function_calling_config": {"mode": "AUTO"}}
    assert config["system_instruction"] == "s"
    assert request["contents"] == [
        {"role": "user", "parts": [{"text": "weather?"}]},
        {
            "role": "model",
            "parts": [
                {"text": "checking"},
                {"function_call": {"name": "get_weather", "args": {"city": "Oslo"}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                # Name looked up from the call earlier in the conversation.
                {"function_response": {"name": "get_weather", "response": {"result": "12C"}}},
                {"function_response": {"name": "get_time", "response": {"error": "no clock"}}},
            ],
        },
    ]
    (got,) = response.tool_calls
    assert got.name == "get_weather" and got.arguments == {"city": "Oslo"}
    assert got.id.startswith("call_")
    assert response.text == "" and response.metadata == {"tool_calling": "native"}


def test_tool_choice_modes(make_provider):
    models = Models(reply=_reply("ok"))
    provider = make_provider(models)
    provider.chat_with_tools([Message.user("q")], [WEATHER, TIME], tool_choice="required")
    assert models.calls[-1]["config"]["tool_config"]["function_calling_config"] == {"mode": "ANY"}
    provider.chat_with_tools([Message.user("q")], [WEATHER, TIME], tool_choice="none")
    assert models.calls[-1]["config"]["tool_config"]["function_calling_config"] == {"mode": "NONE"}
    provider.chat_with_tools([Message.user("q")], [WEATHER, TIME], tool_choice="get_time")
    assert models.calls[-1]["config"]["tool_config"]["function_calling_config"] == {
        "mode": "ANY",
        "allowed_function_names": ["get_time"],
    }


def test_result_without_a_resolvable_name_is_rejected(make_provider):
    models = Models(reply=_reply("ok"))
    with pytest.raises(UnsupportedContentError, match="no name"):
        make_provider(models).chat_with_tools(
            [Message.user("q"), Message.tool_result("unknown", "x")], [WEATHER]
        )
    assert models.calls == []


def test_empty_reply_without_calls_is_still_an_error(make_provider):
    with pytest.raises(LLMError):
        make_provider(Models(reply=_reply(None))).chat_with_tools([Message.user("q")], [WEATHER])


