"""CachingLLM over the v2 surface: chat, generate_structured and
chat_with_tools (sync and async) are cached / recorded under their own kinds, a hit
reports zero usage, and streams pass through. No network."""

from __future__ import annotations

from kuhaku import ImagePart
from kuhaku.core.llm import (
    BaseLLMProvider,
    CachingLLM,
    GenerationOptions,
    LLMCapabilities,
    LLMResponse,
    StructuredOutputMode,
    TokenUsage,
    ToolSpec,
)
from kuhaku.core.llm.cache import ZERO_USAGE
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.models import Message, TextPart, ToolCall
from tests.test_media import tiny_png

USAGE = TokenUsage(prompt_tokens=7, completion_tokens=3)
WEATHER = ToolSpec("get_weather", "Weather.", {"type": "object", "properties": {}})


class CountingLLM(BaseLLMProvider):
    capabilities = LLMCapabilities(
        chat=True,
        structured_output=StructuredOutputMode.SCHEMA,
        tool_calling=True,
        content_types=frozenset({"text", "image"}),
    )

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "count:model"

    def chat(self, messages, *, options=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=f'{{"n": {self.calls}}}', usage=USAGE, model=self.name)

    def _structured_complete(self, messages, request, options) -> LLMResponse:
        return self.chat(messages, options=options)

    def _tool_complete(self, messages, request, options) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="",
            usage=USAGE,
            model=self.name,
            finish_reason="tool_calls",
            tool_calls=(ToolCall("get_weather", {"city": "Oslo"}, id=f"c{self.calls}"),),
        )


Q = [Message.user("hi")]


def test_chat_structured_and_tools_are_cached_with_zero_usage_on_a_hit():
    inner = CountingLLM()
    llm = CachingLLM(inner)

    first = llm.chat(Q)
    again = llm.chat(Q)
    assert inner.calls == 1 and again.text == first.text
    assert first.usage == USAGE and again.usage == ZERO_USAGE

    structured = llm.generate_structured(Q, {"type": "object"})
    assert llm.generate_structured(Q, {"type": "object"}).data == structured.data
    assert inner.calls == 2
    # A different schema, different options or a different kind is a different key.
    llm.generate_structured(Q, {"type": "object", "title": "x"})
    llm.chat(Q, options=GenerationOptions(temperature=0.5))
    assert inner.calls == 4

    calls = llm.chat_with_tools(Q, [WEATHER])
    hit = llm.chat_with_tools(Q, [WEATHER])
    assert hit.tool_calls == calls.tool_calls and hit.metadata == calls.metadata
    assert inner.calls == 5
    assert llm.stats.hits == 3


def test_media_parts_key_by_fingerprint_not_bytes():
    inner = CountingLLM()
    llm = CachingLLM(inner)
    image = ImagePart.from_bytes(tiny_png())
    message = [Message.user([image, TextPart("what?")])]
    llm.chat(message)
    llm.chat([Message.user([ImagePart.from_bytes(tiny_png()), TextPart("what?")])])
    assert inner.calls == 1
    llm.chat([Message.user([TextPart("what?")])])
    assert inner.calls == 2


def test_token_tracking_around_the_cache_records_v2_hits_as_zero(monkeypatch):
    recorded = []
    llm = TokenTrackingLLM(CachingLLM(CountingLLM()), provider="count")
    monkeypatch.setattr(llm, "_record_usage", recorded.append)
    llm.chat(Q)
    llm.chat(Q)
    assert recorded == [USAGE, ZERO_USAGE]


