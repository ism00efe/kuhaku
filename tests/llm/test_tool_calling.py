"""Tool calling: ToolSpec, tool-bearing messages, argument parsing, emulation, the
one-call executor, and the wrappers. Provider wire formats are in
``test_tool_calling_providers.py``. No network."""

from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from kuhaku import ToolCallPart, ToolResultPart, ToolSpec
from kuhaku.core.llm import (
    BaseLLMProvider,
    LegacyLLMAdapter,
    LLMResponse,
    LLMWrapper,
    StructuredOutputError,
    TokenUsage,
    ToolCallingLLMProvider,
    ToolRequest,
    UnsupportedContentError,
    as_tool_calling_provider,
    execute_tool_call,
    tool_result_message,
)
from kuhaku.core.llm.base import LLMCapabilities, StructuredOutputMode
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.llm.tool_calling import (
    render_for_emulation,
    split_tool_message,
    tool_call_from_raw,
)
from kuhaku.core.models import ExecutionResult, Message, TextPart, ToolCall


def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Current weather for a city.

    Not part of the description.
    """

    return f"12{unit} in {city}"


WEATHER = ToolSpec.from_function(get_weather)


# --- ToolSpec -------------------------------------------------------------------------


def test_from_function_builds_schema_name_and_description():
    assert WEATHER.name == "get_weather"
    assert WEATHER.description == "Current weather for a city."
    params = WEATHER.parameters
    assert params["type"] == "object" and "title" not in params
    assert params["required"] == ["city"]
    assert params["properties"]["unit"]["enum"] == ["c", "f"]
    assert params["properties"]["unit"]["default"] == "c"


def test_from_function_overrides_and_rejects_varargs():
    spec = ToolSpec.from_function(get_weather, name="weather", description="d")
    assert (spec.name, spec.description) == ("weather", "d")

    def bad(*args):
        return None

    with pytest.raises(TypeError, match="args"):
        ToolSpec.from_function(bad)

    def positional(a, /):
        return a

    with pytest.raises(TypeError, match="positional-only"):
        ToolSpec.from_function(positional)


def test_from_function_unannotated_parameter_accepts_anything():
    def f(x, y: int = 2):
        return x

    spec = ToolSpec.from_function(f)
    assert spec.parameters["required"] == ["x"]
    assert spec.parameters["properties"]["y"]["type"] == "integer"


def test_from_model():
    class SearchArgs(BaseModel):
        """Search the catalogue."""

        query: str = Field(description="what to look for")
        limit: int = 5

    spec = ToolSpec.from_model(SearchArgs, name="search")
    assert spec.name == "search" and spec.description == "Search the catalogue."
    assert spec.parameters["properties"]["query"]["description"] == "what to look for"
    assert ToolSpec.from_model(SearchArgs).name == "SearchArgs"


def test_toolspec_validates_name_and_parameters():
    with pytest.raises(ValueError, match="name"):
        ToolSpec("has space")
    with pytest.raises(ValueError, match="name"):
        ToolSpec("x" * 65)
    with pytest.raises(ValueError, match="object"):
        ToolSpec("t", parameters={"type": "string"})
    assert ToolSpec("t").parameters == {"type": "object", "properties": {}}


def test_tool_request_validation():
    other = ToolSpec("other")
    assert ToolRequest.build([WEATHER, other], "other").forced == "other"
    assert ToolRequest.build([WEATHER], "required").forced is None
    with pytest.raises(ValueError, match="tool_choice"):
        ToolRequest.build([WEATHER], "missing")
    with pytest.raises(ValueError, match="empty"):
        ToolRequest.build([], "auto")
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRequest.build([WEATHER, WEATHER])
    with pytest.raises(TypeError):
        ToolRequest.build([get_weather])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        ToolRequest.build(WEATHER)  # type: ignore[arg-type]


# --- argument parsing -------------------------------------------------------------------


def test_tool_call_from_raw_parses_and_flags():
    ok = tool_call_from_raw("f", '{"a": 1}', id="c1")
    assert ok == ToolCall("f", {"a": 1}, id="c1") and ok.valid
    assert tool_call_from_raw("f", {"a": 1}, id="c").arguments == {"a": 1}
    assert tool_call_from_raw("f", "", id="c").arguments == {}
    assert tool_call_from_raw("f", None, id="c").arguments == {}

    bad = tool_call_from_raw("f", '{"a": ', id="c2")
    assert not bad.valid and bad.arguments == {}
    assert bad.raw_arguments == '{"a": ' and "not valid JSON" in bad.arguments_error

    array = tool_call_from_raw("f", "[1]", id="c3")
    assert not array.valid and "list" in array.arguments_error

    generated = tool_call_from_raw("f", {})
    assert generated.id.startswith("call_")


# --- messages ---------------------------------------------------------------------------


def test_tool_result_message_and_tool_calls_property():
    message = Message.tool_result("c1", "12c", name="get_weather")
    assert message.role == "tool"
    assert message.parts == (ToolResultPart("c1", "12c", name="get_weather"),)
    assert message.text == ""

    call = ToolCall("get_weather", {"city": "Oslo"}, id="c1")
    assistant = Message.assistant([TextPart("checking"), ToolCallPart(call)])
    assert assistant.tool_calls == (call,)
    assert assistant.text == "checking"
    assert Message.user("hi").tool_calls == ()


def test_response_message_round_trips_tool_calls():
    call = ToolCall("get_weather", {"city": "Oslo"}, id="c1")
    assert LLMResponse(text="hi").message == Message.assistant("hi")
    message = LLMResponse(text="", tool_calls=(call,)).message
    assert message.role == "assistant" and message.parts == (ToolCallPart(call),)
    with_text = LLMResponse(text="ok", tool_calls=(call,)).message
    assert with_text.parts == (TextPart("ok"), ToolCallPart(call))


def test_split_tool_message_rejects_misplaced_parts():
    call = ToolCall("f", id="c")
    with pytest.raises(UnsupportedContentError, match="assistant"):
        split_tool_message(Message.user([ToolCallPart(call)]), provider="p")
    with pytest.raises(UnsupportedContentError, match="tool message"):
        split_tool_message(Message("tool", "plain text"), provider="p")
    with pytest.raises(UnsupportedContentError, match="tool message"):
        split_tool_message(Message.user([ToolResultPart("c", "x")]), provider="p")
    with pytest.raises(UnsupportedContentError, match="role"):
        split_tool_message(Message("critic", "x"), provider="p")


def test_render_for_emulation_turns_tool_turns_into_text():
    call = ToolCall("get_weather", {"city": "Oslo"}, id="c1")
    rendered = render_for_emulation(
        [
            Message.user("weather?"),
            Message.assistant([ToolCallPart(call)]),
            Message.tool_result("c1", "12c"),
            Message.tool_result("c2", "boom", name="other", is_error=True),
        ],
        provider="p",
    )
    assert rendered[0] == Message.user("weather?")
    assert json.loads(rendered[1].text) == {
        "tool_calls": [{"id": "c1", "name": "get_weather", "arguments": {"city": "Oslo"}}]
    }
    assert rendered[2].role == "user"
    assert "Tool result for call c1 (get_weather):\n12c" == rendered[2].text
    assert rendered[3].text.startswith("Tool error for call c2 (other)")


# --- emulation --------------------------------------------------------------------------


class ScriptedV1:
    """A v1 provider replaying fixed replies and recording prompts."""

    name = "scripted:v1"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_legacy_adapter_emulates_tool_calls():
    v1 = ScriptedV1('{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Oslo"}}]}')
    llm = LegacyLLMAdapter(v1)
    assert not llm.capabilities.tool_calling
    response = llm.chat_with_tools([Message.user("weather in Oslo?")], [WEATHER])
    assert response.metadata == {"tool_calling": "emulated"}
    assert response.finish_reason == "tool_calls" and response.text == ""
    (call,) = response.tool_calls
    assert call.name == "get_weather" and call.arguments == {"city": "Oslo"}
    assert call.id.startswith("call_")
    system, _ = v1.prompts[0]
    assert "get_weather" in system and '"tool_calls"' in system


def test_emulation_answer_and_prose_fallback():
    llm = LegacyLLMAdapter(ScriptedV1('{"answer": "It is sunny."}', "Just prose.", "Still prose."))
    response = llm.chat_with_tools([Message.user("q")], [WEATHER])
    assert response.text == "It is sunny." and response.tool_calls == ()
    # Unparseable even after the repair turn: taken as a prose answer under "auto".
    response = llm.chat_with_tools([Message.user("q")], [WEATHER])
    assert response.text == "Still prose." and response.tool_calls == ()


def test_emulation_required_raises_when_no_json():
    llm = LegacyLLMAdapter(ScriptedV1("nope", "still nope"))
    with pytest.raises(StructuredOutputError):
        llm.chat_with_tools([Message.user("q")], [WEATHER], tool_choice="required")


def test_emulation_named_choice_and_none():
    v1 = ScriptedV1('{"tool_calls": [{"name": "get_weather", "arguments": "{bad"}]}', "plain")
    llm = LegacyLLMAdapter(v1)
    response = llm.chat_with_tools([Message.user("q")], [WEATHER], tool_choice="get_weather")
    (call,) = response.tool_calls
    assert not call.valid and call.raw_arguments == "{bad"
    assert 'must call the tool "get_weather"' in v1.prompts[0][0]

    response = llm.chat_with_tools(
        [Message.user("q"), response.message, Message.tool_result(call.id, "err", is_error=True)],
        [WEATHER],
        tool_choice="none",
    )
    assert response.text == "plain" and response.metadata == {"tool_calling": "emulated"}
    assert "get_weather" in v1.prompts[1][1] and "Tool error" in v1.prompts[1][1]


class SchemaProvider(BaseLLMProvider):
    """A provider with native structured output but no tool calling."""

    capabilities = LLMCapabilities(chat=True, structured_output=StructuredOutputMode.SCHEMA)

    def __init__(self, reply: str):
        self.reply = reply
        self.schemas: list[dict] = []

    @property
    def name(self) -> str:
        return "schema:v1"

    def chat(self, messages, *, options=None):
        return LLMResponse(text=self.reply, model=self.name)

    def _structured_complete(self, messages, request, options):
        self.schemas.append(request.schema)
        return LLMResponse(text=self.reply, usage=TokenUsage(3, 2), model=self.name)


def test_emulation_uses_native_structured_output_with_the_reply_schema():
    llm = SchemaProvider('{"tool_calls": [{"name": "get_weather", "arguments": {}}]}')
    response = llm.chat_with_tools([Message.user("q")], [WEATHER], tool_choice="required")
    assert response.usage == TokenUsage(3, 2)
    schema = llm.schemas[0]
    assert schema["required"] == ["tool_calls"]
    assert schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"] == [
        "get_weather"
    ]


# --- wrappers -------------------------------------------------------------------------


class ThirdPartyChat:
    """A v2 provider written before tool calling: not a BaseLLMProvider subclass."""

    name = "third:v1"
    capabilities = LLMCapabilities(chat=True)

    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, system, user):
        return self.reply

    def chat(self, messages, *, options=None):
        return LLMResponse(text=self.reply, usage=TokenUsage(1, 1), model=self.name)

    def generate_structured(self, messages, json_schema, **kwargs):
        return LegacyLLMAdapter(self).generate_structured(messages, json_schema, **kwargs)

    def stream(self, messages, *, options=None):
        raise NotImplementedError

    async def agenerate(self, system, user):
        return self.reply

    async def achat(self, messages, *, options=None):
        return self.chat(messages)

    async def agenerate_structured(self, messages, json_schema, **kwargs):
        return self.generate_structured(messages, json_schema, **kwargs)

    def astream(self, messages, *, options=None):
        raise NotImplementedError


def test_as_tool_calling_provider_lifts_a_pre_tool_v2_provider():
    third = ThirdPartyChat('{"tool_calls": [{"name": "get_weather", "arguments": {}}]}')
    assert not isinstance(third, ToolCallingLLMProvider)
    lifted = as_tool_calling_provider(third)
    assert isinstance(lifted, LLMWrapper) and isinstance(lifted, ToolCallingLLMProvider)
    response = lifted.chat_with_tools([Message.user("q")], [WEATHER])
    assert response.tool_calls[0].name == "get_weather"
    assert response.metadata == {"tool_calling": "emulated"}

    native = LegacyLLMAdapter(ScriptedV1())
    assert as_tool_calling_provider(native) is native


class RecordingTracker(TokenTrackingLLM):
    def __init__(self, wrapped):
        super().__init__(wrapped, provider="scripted")
        self.recorded: list[TokenUsage | None] = []

    def _record_usage(self, usage):
        self.recorded.append(usage)


def test_token_tracking_records_tool_calls_and_passes_them_through():
    llm = SchemaProvider('{"tool_calls": [{"name": "get_weather", "arguments": {}}]}')
    tracker = RecordingTracker(llm)
    response = tracker.chat_with_tools([Message.user("q")], [WEATHER], tool_choice="required")
    assert response.tool_calls[0].name == "get_weather"
    assert tracker.recorded == [TokenUsage(3, 2)]


# --- executing one call -------------------------------------------------------------------


def test_execute_tool_call_success_and_failures():
    def boom(**_):
        raise RuntimeError("down")

    registry = {"get_weather": get_weather, "boom": boom}
    ok = execute_tool_call(ToolCall("get_weather", {"city": "Oslo"}, id="c1"), registry)
    assert ok == ExecutionResult(
        output="12c in Oslo", metadata={"tool_call_id": "c1", "tool": "get_weather"}
    )

    failed = execute_tool_call(ToolCall("boom", id="c2"), registry)
    assert not failed.success and failed.error == "RuntimeError: down"

    unknown = execute_tool_call(ToolCall("nope", id="c3"), registry)
    assert not unknown.success and "Unknown tool" in unknown.error

    wrong_args = execute_tool_call(ToolCall("get_weather", {"town": "x"}, id="c4"), registry)
    assert not wrong_args.success and wrong_args.error.startswith("TypeError")

    flagged = execute_tool_call(tool_call_from_raw("get_weather", "{", id="c5"), registry)
    assert not flagged.success and "Malformed arguments" in flagged.error


def test_execute_tool_call_reports_async_callables():
    async def slow(x: int) -> int:
        return x * 2

    result = execute_tool_call(ToolCall("slow", {"x": 2}, id="c"), {"slow": slow})
    assert not result.success and "synchronous" in result.error



def test_tool_result_message_encodes_output_and_errors():
    call = ToolCall("f", id="c1")
    assert tool_result_message(call, ExecutionResult("text")) == Message.tool_result(
        "c1", "text", name="f"
    )
    data = tool_result_message(call, ExecutionResult({"n": 1, "when": object}))
    assert json.loads(data.parts[0].content)["n"] == 1
    error = tool_result_message(call, ExecutionResult(None, success=False, error="boom"))
    assert error.parts[0] == ToolResultPart("c1", "boom", name="f", is_error=True)
