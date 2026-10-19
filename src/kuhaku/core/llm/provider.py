"""The v2 provider building blocks: a base class, and an adapter for v1 providers.

:class:`BaseLLMProvider` implements the whole :class:`~kuhaku.core.llm.base.ChatLLMProvider`
surface on top of one required method, ``chat``. A subclass adds native features by
overriding a hook and declaring it in ``capabilities``:

- ``_structured_complete(messages, request, options)`` -- one turn in the provider's
  native structured-output mode (declare ``structured_output``);
- ``_tool_complete(messages, request, options)`` -- native tool calling (declare
  ``tool_calling``).

Without them, structured output is emulated by prompt + parse + bounded repair, and
tool calling goes through structured output
(:func:`~kuhaku.core.llm.tool_calling.run_emulated_tools`).

:func:`as_chat_provider` lifts a v1 provider (``generate`` + ``name`` only) to the v2
surface: a conversation is flattened into one system + user prompt.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from tenacity import RetryCallState, retry_if_exception, wait_exponential
from tenacity.wait import wait_base

from ..media import MediaPart
from ..models import Message, TextPart
from ..retry import call_with_retry
from .base import (
    ChatLLMProvider,
    GenerationOptions,
    LLMCapabilities,
    LLMError,
    LLMProvider,
    LLMResponse,
    StructuredResponse,
    UnsupportedContentError,
    counts_as_breaker_failure,
    is_retryable_request_exception,
)
from .structured import StructuredRequest, run_structured
from .tool_calling import ToolRequest, ToolSpec, run_emulated_tools

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Roles every built-in provider can send through chat(). "tool" (a tool result) is
# only sent by the chat_with_tools path, whose encoders know where it goes.
CHAT_ROLES = frozenset({"system", "user", "assistant"})

# Images and documents travel only in user turns: no built-in API takes them in a
# system prompt, and most reject them in an assistant turn.
MEDIA_ROLES = frozenset({"user"})


def normalize_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    if isinstance(messages, (str, Message)):
        raise TypeError("messages must be a sequence of Message, not a single value.")
    normalized = tuple(messages)
    if not normalized:
        raise LLMError("messages must not be empty.")
    for message in normalized:
        if not isinstance(message, Message):
            raise TypeError(
                f"messages must contain Message objects; got {type(message).__name__}."
            )
    return normalized


def message_text(message: Message, *, provider: str) -> str:
    """The message as plain text, or :class:`UnsupportedContentError` if it is not
    text-only or its role is not one a built-in provider sends."""

    if message.role not in CHAT_ROLES:
        raise UnsupportedContentError(
            f"{provider} cannot send a message with role '{message.role}'; "
            f"supported roles: {', '.join(sorted(CHAT_ROLES))}."
        )
    unsupported = sorted({part.type for part in message.parts if not isinstance(part, TextPart)})
    if unsupported:
        raise UnsupportedContentError(
            f"{provider} cannot send content of type {', '.join(unsupported)}; "
            f"only text is supported."
        )
    return message.text


def is_text_only(message: Message) -> bool:
    """True when every part is text -- such a message is encoded exactly as before
    content parts existed, so text-only payloads never change."""

    return all(isinstance(part, TextPart) for part in message.parts)


def check_media(message: Message, *, provider: str, content_types: frozenset[str]) -> None:
    """Raise :class:`UnsupportedContentError` unless every part of ``message`` is text
    or a :class:`~kuhaku.core.media.MediaPart` whose ``type`` is in ``content_types``,
    in a role that may carry media. Logs what is sent -- type, media type and size,
    never the content."""

    if message.role not in CHAT_ROLES:
        raise UnsupportedContentError(
            f"{provider} cannot send a message with role '{message.role}'; "
            f"supported roles: {', '.join(sorted(CHAT_ROLES))}."
        )
    media: list[MediaPart] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            continue
        if not isinstance(part, MediaPart):
            raise UnsupportedContentError(
                f"{provider} cannot send content of type {part.type or '?'} "
                f"({type(part).__name__})."
            )
        if part.type not in content_types:
            raise UnsupportedContentError(
                f"{provider} cannot send {part.type} content; it is configured for "
                f"{', '.join(sorted(content_types))}. If the model does accept "
                f"{part.type} input, pass content_types= to the provider or set "
                f"KUHAKU_LLM_CONTENT_TYPES."
            )
        if message.role not in MEDIA_ROLES:
            raise UnsupportedContentError(
                f"{provider} cannot send {part.type} content in a '{message.role}' "
                f"message; images and documents go in user messages."
            )
        media.append(part)
    if media:
        logger.debug(
            "llm media content",
            extra={"provider": provider, "parts": [part.describe() for part in media]},
        )


def http_url(part: MediaPart, *, provider: str) -> str:
    """``part.url``, or :class:`UnsupportedContentError` when it is not http(s) --
    ``gs://`` is for Vertex AI, which reads Cloud Storage itself."""

    url = part.url or ""
    if not url.lower().startswith(("http://", "https://")):
        raise UnsupportedContentError(
            f"{provider} can only fetch http(s) URLs, not {url.split(':', 1)[0]}:// ones; "
            f"pass the {part.type}'s bytes instead."
        )
    return url


def resolve_content_types(
    configured: str | Iterable[str] | None,
    *,
    default: frozenset[str],
    encodable: frozenset[str],
    provider: str,
) -> frozenset[str]:
    """The content types a provider instance declares: ``configured`` when given (a
    comma-separated string or an iterable; text is always included), else ``default``.

    Configuring a type the provider has no encoding for is an error now rather than a
    rejected request later.
    """

    if configured is None:
        return default
    items = configured.split(",") if isinstance(configured, str) else configured
    types = frozenset(item.strip().lower() for item in items if item.strip()) | {"text"}
    unknown = types - encodable
    if unknown:
        raise LLMError(
            f"{provider} cannot send content of type {', '.join(sorted(unknown))}; "
            f"it can encode: {', '.join(sorted(encodable))}."
        )
    return types


def split_system(
    messages: Sequence[Message], *, provider: str
) -> tuple[str | None, list[Message]]:
    """``(system text or None, the other messages)`` -- for APIs that take the system
    prompt as a separate field. Several system messages are joined with a blank line."""

    system = [message_text(m, provider=provider) for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    return ("\n\n".join(system) if system else None), rest


def flatten(messages: Sequence[Message], *, provider: str) -> tuple[str, str]:
    """A conversation as one ``(system, user)`` pair, for a v1 ``generate``.

    A single user turn passes through unchanged; a longer conversation becomes a
    role-labelled transcript, which is the best a single-prompt API can do.
    """

    system, rest = split_system(messages, provider=provider)
    if len(rest) == 1 and rest[0].role == "user":
        user = message_text(rest[0], provider=provider)
    else:
        user = "\n\n".join(
            f"{m.role.upper()}: {message_text(m, provider=provider)}" for m in rest
        )
    return system or "", user


_NATIVE_TOOLS = {"tool_calling": "native"}


class BaseLLMProvider:
    """Base class for v2 providers. See the module docstring."""

    capabilities: LLMCapabilities = LLMCapabilities()

    @property
    def name(self) -> str:
        raise NotImplementedError

    # --- the one required method -----------------------------------------------------

    def chat(
        self, messages: Sequence[Message], *, options: GenerationOptions | None = None
    ) -> LLMResponse:
        raise NotImplementedError(f"{type(self).__name__} must implement chat().")

    # --- optional native hooks --------------------------------------------------------

    def _structured_complete(
        self,
        messages: tuple[Message, ...],
        request: StructuredRequest,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        raise NotImplementedError(
            f"{type(self).__name__} declares structured_output="
            f"{self.capabilities.structured_output.value!r} but does not implement "
            f"_structured_complete()."
        )

    def _tool_complete(
        self,
        messages: tuple[Message, ...],
        request: ToolRequest,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        raise NotImplementedError(
            f"{type(self).__name__} declares tool_calling=True but does not implement "
            f"_tool_complete()."
        )

    def _tool_response(
        self,
        messages: tuple[Message, ...],
        request: ToolRequest,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        if self.capabilities.tool_calling:
            response = self._tool_complete(messages, request, options)
            return replace(response, metadata={**response.metadata, **_NATIVE_TOOLS})
        return run_emulated_tools(
            chat_turn=lambda turn: self.chat(turn, options=options),
            structured_turn=lambda turn, schema: self.generate_structured(
                turn, schema, name="tool_calls", options=options
            ),
            messages=messages,
            request=request,
            provider=self.name,
        )

    # --- the v2 surface, derived ------------------------------------------------------

    def generate(self, system: str, user: str) -> str:
        return self.chat([Message.system(system), Message.user(user)]).text

    def generate_structured(
        self,
        messages: Sequence[Message],
        json_schema: dict[str, Any],
        *,
        name: str = "response",
        options: GenerationOptions | None = None,
        max_repair_attempts: int = 1,
    ) -> StructuredResponse:
        return run_structured(
            mode=self.capabilities.structured_output,
            native_turn=lambda turn, request: self._structured_complete(turn, request, options),
            chat_turn=lambda turn: self.chat(turn, options=options),
            messages=normalize_messages(messages),
            json_schema=json_schema,
            name=name,
            max_repair_attempts=max_repair_attempts,
        )

    # --- tool calling, derived ----------------------------------------------------------

    def chat_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        tool_choice: str = "auto",
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        request = ToolRequest.build(tools, tool_choice)
        return self._tool_response(normalize_messages(messages), request, options)


# The longest Retry-After a provider's own retry will sleep for. A longer hint (a
# per-day quota, say) is not slept through inside one call: the call fails with an
# LLMRateLimitError carrying the hint, for a RateLimitedLLM or the caller to act on.
RETRY_AFTER_CAP_SECONDS = 60.0


def _retry_after(exc: BaseException | None) -> float | None:
    if exc is None:
        return None
    from .limits.rate_limit import retry_after_seconds

    return retry_after_seconds(exc)


class wait_retry_after(wait_base):
    """Exponential backoff, but never shorter than the provider's ``Retry-After``."""

    def __init__(self, fallback: wait_base, *, cap: float = RETRY_AFTER_CAP_SECONDS) -> None:
        self._fallback = fallback
        self._cap = cap

    def __call__(self, retry_state: RetryCallState) -> float:
        backoff = self._fallback(retry_state)
        outcome = retry_state.outcome
        hint = _retry_after(outcome.exception() if outcome is not None else None)
        if hint is None:
            return backoff
        return max(backoff, min(hint, self._cap))


class ResilientCallMixin:
    """Circuit breaker -> retry -> call, as every built-in provider wires it.

    Expects the ``_circuit_breaker`` / ``_retry_*`` attributes the providers already
    set in ``__init__``. Exceptions pass through unchanged (``CircuitOpenError`` or the
    call's own); each provider maps them to ``LLMError`` with its own wording.

    A 429's ``Retry-After`` sets the wait before the next attempt (up to
    :data:`RETRY_AFTER_CAP_SECONDS`; a longer hint is not retried here at all). Only
    failures that say the provider is unhealthy count toward the circuit breaker --
    see :func:`~kuhaku.core.llm.base.counts_as_breaker_failure`.
    """

    _circuit_breaker: Any
    _retry_enabled: bool
    _retry_max_attempts: int
    _retry_backoff_base_seconds: float
    _retry_backoff_max_seconds: float

    def _resilient(
        self,
        fn: Callable[[], T],
        *,
        retryable: Callable[[BaseException], bool] = is_retryable_request_exception,
    ) -> T:
        def should_retry(exc: BaseException) -> bool:
            if not retryable(exc):
                return False
            hint = _retry_after(exc)
            return hint is None or hint <= RETRY_AFTER_CAP_SECONDS

        def with_retry() -> T:
            return call_with_retry(
                fn,
                service="llm",
                enabled=self._retry_enabled,
                max_attempts=self._retry_max_attempts,
                wait=wait_retry_after(
                    wait_exponential(
                        multiplier=self._retry_backoff_base_seconds,
                        max=self._retry_backoff_max_seconds,
                    )
                ),
                retry=retry_if_exception(should_retry),
            )

        if self._circuit_breaker is not None:
            return self._circuit_breaker.call(with_retry, is_failure=counts_as_breaker_failure)
        return with_retry()


def option(value: T | None, default: T) -> T:
    return default if value is None else value


class LegacyLLMAdapter(BaseLLMProvider):
    """The v2 surface over a v1 provider, by emulation.

    ``generate`` goes straight to the wrapped provider, so v1 behaviour is untouched.
    ``chat`` flattens the conversation (see :func:`flatten`); per-call ``options``
    cannot be passed to a v1 provider and are ignored. Usage comes from the wrapped
    provider's ``last_usage``, when it has one.
    """

    def __init__(self, wrapped: LLMProvider) -> None:
        self._wrapped = wrapped

    @property
    def wrapped(self) -> LLMProvider:
        return self._wrapped

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def last_usage(self) -> Any:
        return getattr(self._wrapped, "last_usage", None)

    def generate(self, system: str, user: str) -> str:
        return self._wrapped.generate(system, user)

    def chat(
        self, messages: Sequence[Message], *, options: GenerationOptions | None = None
    ) -> LLMResponse:
        system, user = flatten(normalize_messages(messages), provider=self.name)
        text = self._wrapped.generate(system, user)
        return LLMResponse(text=text, usage=self.last_usage, model=self.name)


def as_chat_provider(llm: LLMProvider) -> ChatLLMProvider:
    """``llm`` itself if it already implements the v2 surface, else a
    :class:`LegacyLLMAdapter` around it."""

    if isinstance(llm, ChatLLMProvider):
        return llm
    return LegacyLLMAdapter(llm)


def capabilities_of(llm: LLMProvider) -> LLMCapabilities:
    """``llm.capabilities``, or the emulated defaults for a v1 provider."""

    capabilities = getattr(llm, "capabilities", None)
    return capabilities if isinstance(capabilities, LLMCapabilities) else LLMCapabilities()
