"""``LLMWrapper``: the base for decorators around an ``LLMProvider``.

A wrapper (token tracking today; caching, budgets, rate limits later) intercepts calls
on their way to a provider. This class delegates every v1 and v2 method to the wrapped
provider so a subclass overrides only the methods it cares about and still presents
the full v2 surface -- including when it wraps a v1-only provider, which is lifted with
:func:`~kuhaku.core.llm.provider.as_chat_provider`.

``generate`` is delegated to the wrapped provider's own ``generate`` rather than routed
through ``chat``, so a wrapper never changes v1 behaviour. The flip side: a subclass
that must see every call overrides ``generate`` *and* the v2 methods; overriding
``chat`` alone does not intercept ``generate``.

``generate(system, user)`` and ``chat([Message.system(system), Message.user(user)])``
with no options are one *logical* call (:func:`generate_call` recognises the second
form). Wrappers that key, meter or record calls treat them alike --
:func:`generate_response` runs either form and returns an ``LLMResponse`` carrying that
call's own usage, so neither the sync/async split nor the v1/v2 split changes a cache
key or what gets counted.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..models import Message, TextPart
from .base import (
    ChatLLMProvider,
    GenerationOptions,
    LLMCapabilities,
    LLMProvider,
    LLMResponse,
    StructuredResponse,
    TokenUsage,
    ToolCallingLLMProvider,
)
from .provider import LegacyLLMAdapter, as_chat_provider, normalize_messages
from .tool_calling import ToolRequest, ToolSpec, run_emulated_tools

logger = logging.getLogger(__name__)


def generate_call(
    messages: Sequence[Message], options: GenerationOptions | None
) -> tuple[str, str] | None:
    """``(system, user)`` when ``chat(messages, options=options)`` is exactly what
    ``generate(system, user)`` sends -- one plain-text system turn, one plain-text user
    turn, no per-call options -- else ``None``."""

    if options is not None and (options.temperature is not None or options.max_tokens is not None):
        return None
    if isinstance(messages, (str, Message)):
        return None
    try:
        turns = tuple(messages)
    except TypeError:
        return None
    if len(turns) != 2 or not all(isinstance(m, Message) for m in turns):
        return None
    system, user = turns
    if system.role != "system" or user.role != "user":
        return None
    if not all(len(m.parts) == 1 and isinstance(m.parts[0], TextPart) for m in turns):
        return None
    return system.text, user.text


def _fresh_last_usage(llm: object, before: object) -> TokenUsage | None:
    # Providers assign a new TokenUsage per report and never clear it: an unchanged
    # object after the call is the previous call's numbers, not this one's.
    after = getattr(llm, "last_usage", None)
    if after is before or not isinstance(after, TokenUsage):
        return None
    return after


def generate_response(llm: LLMProvider, system: str, user: str) -> LLMResponse:
    """Run ``generate(system, user)`` on ``llm`` and return it as an ``LLMResponse``
    whose ``usage`` belongs to this call.

    A v2 provider (or wrapper) is asked through ``chat``, whose response carries its
    own usage -- right under concurrency, where the shared ``last_usage`` slot is not.
    A v1 provider is asked through ``generate``; its ``last_usage`` is used only when the
    call replaced it, so a provider that reported nothing is not booked at the previous
    call's numbers.
    """

    if isinstance(llm, LegacyLLMAdapter):
        llm = llm.wrapped
    if isinstance(llm, ChatLLMProvider):
        return llm.chat([Message.system(system), Message.user(user)])
    before = getattr(llm, "last_usage", None)
    text = llm.generate(system, user)
    return LLMResponse(text=text, usage=_fresh_last_usage(llm, before), model=llm.name)


class LLMWrapper:
    def __init__(self, wrapped: LLMProvider) -> None:
        self._wrapped = wrapped
        self._inner: ChatLLMProvider = as_chat_provider(wrapped)

    @property
    def wrapped(self) -> LLMProvider:
        return self._wrapped

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._inner.capabilities

    @property
    def last_usage(self) -> TokenUsage | None:
        """The wrapped provider's ``last_usage``, so a caller that totals usage per
        request (a tool reporting tokens on its result) can read it through any stack
        of wrappers as it would from the provider itself."""

        return getattr(self._wrapped, "last_usage", None)

    def generate(self, system: str, user: str) -> str:
        return self._wrapped.generate(system, user)

    def chat(
        self, messages: Sequence[Message], *, options: GenerationOptions | None = None
    ) -> LLMResponse:
        inner = self._inner
        if isinstance(inner, LegacyLLMAdapter):
            # A v1 provider's adapter reports last_usage unchecked; for the call it
            # can answer natively, go straight to its generate with a freshness check.
            call = generate_call(messages, options)
            if call is not None:
                return generate_response(inner.wrapped, *call)
        return inner.chat(messages, options=options)

    def generate_structured(
        self,
        messages: Sequence[Message],
        json_schema: dict[str, Any],
        *,
        name: str = "response",
        options: GenerationOptions | None = None,
        max_repair_attempts: int = 1,
    ) -> StructuredResponse:
        return self._inner.generate_structured(
            messages,
            json_schema,
            name=name,
            options=options,
            max_repair_attempts=max_repair_attempts,
        )

    # --- tool calling -----------------------------------------------------------------
    # Delegated when the wrapped provider has tool calling (every built-in provider and
    # BaseLLMProvider subclass does); emulated on its chat/generate_structured
    # otherwise, so a wrapper always offers the tool-calling surface.

    def chat_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        tool_choice: str = "auto",
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        if isinstance(self._inner, ToolCallingLLMProvider):
            return self._inner.chat_with_tools(
                messages, tools, tool_choice=tool_choice, options=options
            )
        return self._emulated(messages, tools, tool_choice, options)

    def _emulated(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: str,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        inner = self._inner
        return run_emulated_tools(
            chat_turn=lambda turn: inner.chat(turn, options=options),
            structured_turn=lambda turn, schema: inner.generate_structured(
                turn, schema, name="tool_calls", options=options
            ),
            messages=normalize_messages(messages),
            request=ToolRequest.build(tools, tool_choice),
            provider=self.name,
        )


def as_tool_calling_provider(llm: LLMProvider) -> ToolCallingLLMProvider:
    """``llm`` itself if it already has the tool-calling surface, else an
    :class:`LLMWrapper` around it (tool calling emulated through structured output)."""

    if isinstance(llm, ToolCallingLLMProvider):
        return llm
    return LLMWrapper(llm)
