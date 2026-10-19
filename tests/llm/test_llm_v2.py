"""Tests for the v2 LLM surface that do not depend on a vendor: messages, the base
class, the v1 adapter, structured-output emulation, streams, async, and wrappers.

Entirely offline: providers here are in-memory fakes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from kuhaku.core.llm import (
    BaseLLMProvider,
    ChatLLMProvider,
    GenerationOptions,
    LegacyLLMAdapter,
    LLMCapabilities,
    LLMError,
    LLMResponse,
    LLMWrapper,
    StructuredOutputError,
    StructuredOutputMode,
    TokenUsage,
    UnsupportedContentError,
    as_chat_provider,
    capabilities_of,
)
from kuhaku.core.llm.structured import parse_json, schema_name
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.models import ContentPart, Message, TextPart
from kuhaku.core.observability.metrics import LLM_TOKENS_TOTAL
from kuhaku.core.observability.tracing import GEN_AI_SYSTEM, GEN_AI_TOKEN_TYPE
from tests.conftest import prometheus_counter_value as _counter_value


class V1Fake:
    """A v1-only provider: generate + name, optionally a last_usage side channel."""

    def __init__(self, *replies: str, usage: TokenUsage | None = None) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []
        self.last_usage = usage

    @property
    def name(self) -> str:
        return "v1:fake"

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.replies.pop(0)


_USAGE = TokenUsage(prompt_tokens=3, completion_tokens=2)


class ChatFake(BaseLLMProvider):
    """A v2 provider built on BaseLLMProvider, replaying scripted replies."""

    def __init__(
        self,
        *replies: str,
        mode: StructuredOutputMode = StructuredOutputMode.NONE,
        usage: TokenUsage | None = _USAGE,
    ) -> None:
        self.replies = list(replies)
        self.usage = usage
        self.chats: list[tuple[Message, ...]] = []
        self.native: list[tuple[tuple[Message, ...], object]] = []
        self.capabilities = LLMCapabilities(chat=True, structured_output=mode)

    @property
    def name(self) -> str:
        return "chat:fake"

    def chat(self, messages: Sequence[Message], *, options=None) -> LLMResponse:
        self.chats.append(tuple(messages))
        return LLMResponse(text=self.replies.pop(0), usage=self.usage, model=self.name)

    def _structured_complete(self, messages, request, options) -> LLMResponse:
        self.native.append((messages, request))
        return LLMResponse(text=self.replies.pop(0), usage=self.usage, model=self.name)


class _ImagePart(ContentPart):
    type = "image"


# --- Message -------------------------------------------------------------------------


def test_message_string_content_is_unchanged():
    message = Message("user", "hi")
    assert message.content == "hi"
    assert message.text == "hi"
    assert message.parts == (TextPart("hi"),)


def test_message_parts_are_normalized_to_a_tuple_and_joined_for_text():
    message = Message.user([TextPart("a"), TextPart("b")])
    assert message.content == (TextPart("a"), TextPart("b"))
    assert message.text == "ab"
    assert message.role == "user"


def test_message_rejects_non_part_items():
    with pytest.raises(TypeError):
        Message("user", ["plain string inside a list"])  # type: ignore[list-item]


def test_message_text_skips_non_text_parts():
    message = Message.user([TextPart("caption"), _ImagePart()])
    assert message.text == "caption"
    assert [part.type for part in message.parts] == ["text", "image"]


def test_message_constructors_set_role():
    assert Message.system("s").role == "system"
    assert Message.assistant("a").role == "assistant"


# --- BaseLLMProvider ---------------------------------------------------------------------


def test_base_generate_is_chat_with_system_and_user():
    llm = ChatFake("answer")
    assert llm.generate("sys", "usr") == "answer"
    assert llm.chats == [(Message.system("sys"), Message.user("usr"))]


def test_base_provider_satisfies_both_protocols():
    from kuhaku.core.llm import LLMProvider

    llm = ChatFake("x")
    assert isinstance(llm, LLMProvider)
    assert isinstance(llm, ChatLLMProvider)
    assert as_chat_provider(llm) is llm


def test_chat_rejects_empty_or_wrong_messages():
    llm = ChatFake("x")
    with pytest.raises(LLMError):
        llm.generate_structured([], {"type": "object"})


# --- structured output -------------------------------------------------------------------

_SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}}


def test_emulated_structured_puts_schema_in_prompt_and_parses():
    llm = ChatFake('```json\n{"n": 1}\n```')
    result = llm.generate_structured([Message.system("be terse"), Message.user("q")], _SCHEMA)
    assert result.data == {"n": 1}
    assert result.native is False
    assert result.attempts == 1
    sent = llm.chats[0]
    assert [m.role for m in sent] == ["system", "system", "user"]
    assert '"integer"' in sent[1].text  # the schema instruction, after the caller's system


def test_emulated_structured_repairs_a_parse_failure_and_sums_usage():
    llm = ChatFake("not json at all", '{"n": 2}')
    result = llm.generate_structured([Message.user("q")], _SCHEMA, max_repair_attempts=1)
    assert result.data == {"n": 2}
    assert result.attempts == 2
    assert result.usage == TokenUsage(prompt_tokens=6, completion_tokens=4)
    repair_turn = llm.chats[1]
    assert repair_turn[-2] == Message.assistant("not json at all")
    assert "could not be parsed" in repair_turn[-1].text


def test_structured_raises_after_repair_budget():
    llm = ChatFake("nope", "still nope")
    with pytest.raises(StructuredOutputError) as info:
        llm.generate_structured([Message.user("q")], _SCHEMA, max_repair_attempts=1)
    assert info.value.attempts == 2
    assert info.value.raw == "still nope"
    assert isinstance(info.value, LLMError)


def test_schema_mode_uses_native_turn_without_prompt_schema():
    llm = ChatFake('{"n": 3}', mode=StructuredOutputMode.SCHEMA)
    result = llm.generate_structured([Message.user("q")], _SCHEMA, name="my schema!")
    assert result.data == {"n": 3}
    assert result.native is True
    messages, request = llm.native[0]
    assert messages == (Message.user("q"),)
    assert request.schema == _SCHEMA
    assert request.name == "my_schema"
    assert llm.chats == []


def test_json_mode_uses_native_turn_with_prompt_schema():
    llm = ChatFake('{"n": 4}', mode=StructuredOutputMode.JSON)
    llm.generate_structured([Message.user("q")], _SCHEMA)
    messages, _ = llm.native[0]
    assert messages[0].role == "system" and '"integer"' in messages[0].text


def test_parse_json_tolerates_prose_and_arrays():
    assert parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert parse_json("[1, 2]") == [1, 2]
    with pytest.raises(StructuredOutputError):
        parse_json("")


def test_schema_name_is_sanitized():
    assert schema_name("Order line / v2") == "Order_line_v2"
    assert schema_name("!!!") == "response"


# --- v1 adapter ----------------------------------------------------------------------


def test_v1_provider_is_lifted_by_emulation():
    v1 = V1Fake("hello", usage=TokenUsage(1, 1))
    llm = as_chat_provider(v1)
    assert isinstance(llm, LegacyLLMAdapter)
    assert capabilities_of(v1) == LLMCapabilities()
    response = llm.chat([Message.system("s"), Message.user("u")])
    assert response.text == "hello"
    assert response.usage == TokenUsage(1, 1)
    assert v1.calls == [("s", "u")]


def test_v1_adapter_flattens_a_conversation_into_a_transcript():
    v1 = V1Fake("ok")
    as_chat_provider(v1).chat(
        [Message.user("first"), Message.assistant("reply"), Message.user("second")]
    )
    system, user = v1.calls[0]
    assert system == ""
    assert user == "USER: first\n\nASSISTANT: reply\n\nUSER: second"


def test_v1_adapter_rejects_unsupported_content():
    llm = as_chat_provider(V1Fake("x"))
    with pytest.raises(UnsupportedContentError):
        llm.chat([Message.user([_ImagePart()])])
    with pytest.raises(UnsupportedContentError):
        llm.chat([Message("tool", "result")])


def test_v1_adapter_structured_output_by_prompt():
    llm = as_chat_provider(V1Fake('{"n": 5}'))
    assert llm.generate_structured([Message.user("q")], _SCHEMA).data == {"n": 5}


# --- async -------------------------------------------------------------------------


# --- streams -----------------------------------------------------------------------


# --- wrappers ----------------------------------------------------------------------


def _tokens(provider: str, kind: str) -> float:
    return _counter_value(LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: provider, GEN_AI_TOKEN_TYPE: kind})


def test_wrapper_delegates_everything_and_passes_capabilities():
    inner = ChatFake("a", '{"n": 1}', mode=StructuredOutputMode.SCHEMA)
    wrapper = LLMWrapper(inner)
    assert wrapper.name == "chat:fake"
    assert wrapper.capabilities is inner.capabilities
    assert isinstance(wrapper, ChatLLMProvider)
    assert wrapper.chat([Message.user("q")]).text == "a"
    assert wrapper.generate_structured([Message.user("q")], _SCHEMA).native is True


def test_wrapper_over_v1_keeps_generate_direct():
    v1 = V1Fake("direct")
    wrapper = LLMWrapper(v1)
    assert wrapper.generate("s", "u") == "direct"
    assert v1.calls == [("s", "u")]
    assert capabilities_of(wrapper) == LLMCapabilities()


def test_token_tracking_records_chat_usage_from_the_response():
    tracker = TokenTrackingLLM(ChatFake("a"), provider="v2chat")
    before = _tokens("v2chat", "input")
    tracker.chat([Message.user("q")])
    assert _tokens("v2chat", "input") == before + 3


def test_token_tracking_records_structured_usage_summed_over_repairs():
    tracker = TokenTrackingLLM(ChatFake("bad", '{"n": 1}'), provider="v2struct")
    before = _tokens("v2struct", "output")
    result = tracker.generate_structured([Message.user("q")], _SCHEMA)
    assert result.attempts == 2
    assert _tokens("v2struct", "output") == before + 4


def test_generation_options_reach_the_provider():
    seen = {}

    class OptionsFake(ChatFake):
        def chat(self, messages, *, options=None):
            seen["options"] = options
            return super().chat(messages, options=options)

    TokenTrackingLLM(OptionsFake("a"), provider="opts").chat(
        [Message.user("q")], options=GenerationOptions(temperature=0.0, max_tokens=5)
    )
    assert seen["options"] == GenerationOptions(temperature=0.0, max_tokens=5)


def test_structured_response_data_is_json_compatible():
    llm = ChatFake('{"items": [1, 2], "ok": true}')
    data = llm.generate_structured([Message.user("q")], _SCHEMA).data
    assert json.loads(json.dumps(data)) == {"items": [1, 2], "ok": True}
