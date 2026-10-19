"""Fakes for the agent tests: a tool-calling model that replays a script and records
every request it was sent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from kuhaku.core.config import Settings
from kuhaku.core.llm import BaseLLMProvider, LLMCapabilities, LLMResponse, TokenUsage
from kuhaku.core.llm.tool_calling import ToolRequest
from kuhaku.core.models import Message, ToolCall

Reply = (
    LLMResponse
    | Exception
    | Callable[[tuple[Message, ...], ToolRequest | None], LLMResponse]
)


def calls(
    *items: tuple[str, dict[str, Any]] | ToolCall, usage: TokenUsage | None = None
) -> LLMResponse:
    """A reply making tool calls: ``("name", {args})`` pairs, ids ``c1``, ``c2``, ..."""

    built = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, ToolCall):
            built.append(item)
        else:
            name, arguments = item
            built.append(ToolCall(name=name, arguments=arguments, id=f"c{index}"))
    return LLMResponse(
        text="", tool_calls=tuple(built), usage=usage, finish_reason="tool_calls"
    )


def answer(text: str, usage: TokenUsage | None = None) -> LLMResponse:
    return LLMResponse(text=text, usage=usage, finish_reason="stop")


class ScriptedToolLLM(BaseLLMProvider):
    """Native tool calling from a script. ``requests`` records, per call, the messages
    and the names of the tools offered (``None`` for a plain ``chat``)."""

    capabilities = LLMCapabilities(chat=True, tool_calling=True)

    def __init__(self, *replies: Reply, name: str = "scripted:tools") -> None:
        self.replies = list(replies)
        self.requests: list[tuple[tuple[Message, ...], tuple[str, ...] | None]] = []
        self.specs: list[Sequence[Any]] = []
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages, *, options=None) -> LLMResponse:
        return self._next(tuple(messages), None)

    def _tool_complete(self, messages, request, options) -> LLMResponse:
        self.specs.append(request.tools)
        return self._next(messages, request)

    def _next(
        self, messages: tuple[Message, ...], request: ToolRequest | None
    ) -> LLMResponse:
        names = tuple(spec.name for spec in request.tools) if request else None
        self.requests.append((messages, names))
        if not self.replies:
            raise AssertionError(f"{self._name} was called more times than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(messages, request)
        return reply


def no_audit() -> Settings:
    return Settings(audit_enabled=False)


def tool_results(messages: Sequence[Message]) -> list[Any]:
    """Every tool-result part in ``messages``, in order."""

    from kuhaku.core.models import ToolResultPart

    return [p for m in messages for p in m.parts if isinstance(p, ToolResultPart)]
