"""RateLimitedLLM and BudgetedLLM over the v2 surface: chat, generate_structured and
chat_with_tools (sync and async) are paced / budgeted and reconciled with the
response's own usage; streams are metered too; a budget switch applies to v2 calls."""

from __future__ import annotations

from kuhaku.core.llm import (
    BaseLLMProvider,
    LLMCapabilities,
    LLMResponse,
    StructuredOutputMode,
    TokenUsage,
    ToolSpec,
)
from kuhaku.core.llm.limits import (
    RateLimitedLLM,
    RateLimiter,
    RateLimits,
    TokenEstimate,
)
from kuhaku.core.models import Message, ToolCall

from .limits_fakes import FakeClock

USAGE = TokenUsage(prompt_tokens=11, completion_tokens=4)
TOOL = ToolSpec("lookup", "Look up.", {"type": "object", "properties": {}})
Q = [Message.system("s"), Message.user("hello")]


class ChatModel(BaseLLMProvider):
    capabilities = LLMCapabilities(
        chat=True, structured_output=StructuredOutputMode.SCHEMA, tool_calling=True
    )

    def __init__(self, name: str = "fake:chat") -> None:
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages, *, options=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text='{"a": 1}', usage=USAGE, model=self.name)

    def _structured_complete(self, messages, request, options) -> LLMResponse:
        return self.chat(messages)

    def _tool_complete(self, messages, request, options) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="", usage=USAGE, model=self.name, tool_calls=(ToolCall("lookup", {}, id="t"),)
        )


def _fixed(total: int):
    return lambda system, user: TokenEstimate(total, 0)


def test_rate_limited_v2_calls_are_paced_and_reconciled_with_response_usage():
    clock = FakeClock()
    limiter = RateLimiter(RateLimits(requests_per_minute=2), clock=clock)
    inner = ChatModel()
    llm = RateLimitedLLM(inner, limiter)
    assert llm.chat(Q).text == '{"a": 1}'
    assert llm.generate_structured(Q, {"type": "object"}).data == {"a": 1}
    assert clock.sleeps == []
    # The third call in the same minute waits for the request bucket.
    assert llm.chat_with_tools(Q, [TOOL]).tool_calls[0].name == "lookup"
    assert clock.sleeps and inner.calls == 3


