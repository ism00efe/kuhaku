"""LLM provider interface.

The RAG engine depends only on this protocol — never on a vendor SDK. Concrete providers
live next to this file and are chosen by ``LLM_PROVIDER`` via the factory in
``__init__.py``.

Two protocols, one a superset of the other:

- :class:`LLMProvider` (v1) -- ``generate(system, user) -> str`` plus ``name``. The
  minimum any provider, fake or wrapper must offer; everything that only needs one
  completion (the RAG engine, the evaluation judges) depends on this and nothing more.
- :class:`ChatLLMProvider` (v2) -- adds a conversation API (``chat``), native
  structured output (``generate_structured``), and :class:`LLMCapabilities` describing
  which of them are native. Every
  built-in provider implements it; :func:`~kuhaku.core.llm.provider.as_chat_provider`
  lifts any v1 provider to it by emulation, so callers can rely on the v2 surface
  without knowing what they were handed.
- :class:`ToolCallingLLMProvider` -- adds ``chat_with_tools``: tool definitions in, :class:`~kuhaku.core.models.ToolCall`\\ s out, natively
  or by emulation (``capabilities.tool_calling``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import requests

from ..models import Message, TextPart, ToolCall, ToolCallPart

if TYPE_CHECKING:
    from .tool_calling import ToolSpec


@dataclass(frozen=True)
class TokenUsage:
    """Token counts from one ``generate()`` call, when the provider's API reports them.

    Not part of the ``LLMProvider`` protocol's return type (``generate()`` still returns
    a plain ``str``): providers instead record their own usage on a
    ``last_usage`` instance attribute, which ``TokenTrackingLLM`` (``llm/token_tracking.py``)
    reads via ``getattr`` after the call completes. This is an addition every provider
    already did the hard part of (all three parse usage out of their response body for
    a log line); it does not change ``generate()``'s signature or return type.

    The v2 methods return usage on the response object instead
    (:attr:`LLMResponse.usage`), which -- unlike ``last_usage`` -- stays correct when
    one provider instance serves concurrent calls.
    """

    prompt_tokens: int | None
    completion_tokens: int | None

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=_add_counts(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=_add_counts(self.completion_tokens, other.completion_tokens),
        )


def _add_counts(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def combine_usage(a: TokenUsage | None, b: TokenUsage | None) -> TokenUsage | None:
    """Sum two optional usages -- for a result that took several API calls."""

    if a is None:
        return b
    if b is None:
        return a
    return a + b


class LLMError(RuntimeError):
    """Raised when a provider call fails (network, auth, bad response)."""


class LLMRequestError(LLMError):
    """A provider call failed with an identifiable cause.

    The typed errors below subclass it, so ``except LLMError`` still catches every one
    of them. ``status_code`` is the HTTP status when there was one; it also lets
    :func:`~kuhaku.core.llm.limits.is_rate_limit_error` recognise a 429 on the error
    itself, not only on its ``__cause__``.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMRateLimitError(LLMRequestError):
    """HTTP 429. ``retry_after`` is the provider's back-off hint in seconds, if any."""

    def __init__(
        self, message: str, *, status_code: int | None = 429, retry_after: float | None = None
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class LLMAuthError(LLMRequestError):
    """HTTP 401/403: the credentials are missing, wrong or lack permission."""


class LLMBadRequestError(LLMRequestError):
    """Any other 4xx: the request itself is wrong (bad parameter, unknown model, too
    long a prompt). Retrying the same request will not help."""


class LLMServerError(LLMRequestError):
    """HTTP 5xx: the provider failed. Usually transient."""


class LLMTimeoutError(LLMRequestError):
    """The request timed out (client-side, or HTTP 408)."""


class LLMConnectionError(LLMRequestError):
    """The provider could not be reached (DNS, refused connection, reset)."""


class LLMRefusalError(LLMError):
    """The model declined to answer (a ``refusal``), so there is no output to return."""


class LLMUnavailableError(LLMError):
    """The LLM provider could not be reached or timed out for this call.

    Raised by :class:`~kuhaku.tools.rag.engine.RAGEngine` when a provider's
    ``generate()`` raises :class:`LLMError` — translated at that boundary so the API
    layer can map it to a 503 without importing provider internals.
    """


class UnsupportedContentError(LLMError):
    """A message carries a role or content part this provider cannot send.

    Raised before any network call. It is how a provider declines content added to
    :class:`~kuhaku.core.models.Message` after it was written (images, tool calls),
    instead of silently dropping it.
    """


class StructuredOutputError(LLMError):
    """``generate_structured`` got no parseable JSON within its repair budget.

    ``raw`` is the last reply text, ``attempts`` the number of model calls made.
    """

    def __init__(self, message: str, *, raw: str = "", attempts: int = 0) -> None:
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


class StructuredOutputMode(str, Enum):
    """How a provider constrains output to a JSON Schema."""

    NONE = "none"
    """No native mode: the schema goes into the prompt and the reply is parsed."""

    JSON = "json"
    """Native JSON mode: the reply is valid JSON, but the schema is only in the prompt."""

    SCHEMA = "schema"
    """Native schema mode: the provider is given the schema itself."""


@dataclass(frozen=True)
class LLMCapabilities:
    """What a provider does natively.

    Every :class:`ChatLLMProvider` method works on every provider -- these flags say
    which are native and which are emulated, so a caller can choose a strategy (the
    extraction tool, for example, only takes its native structured-output path when
    ``structured_output`` is not ``NONE``). The defaults describe a v1 provider
    lifted by emulation.

    ``content_types`` is a seam for later content kinds: today every built-in provider
    reports ``{"text"}``. ``tool_calling`` says the provider's API takes tool
    definitions and returns tool calls itself; without it, ``chat_with_tools`` is
    emulated through structured output and each response says so in
    ``metadata["tool_calling"]``.
    """

    chat: bool = False
    """A conversation is sent as separate turns, not flattened into one prompt."""

    structured_output: StructuredOutputMode = StructuredOutputMode.NONE
    content_types: frozenset[str] = frozenset({"text"})
    tool_calling: bool = False

    @property
    def native_structured_output(self) -> bool:
        return self.structured_output is not StructuredOutputMode.NONE


@dataclass(frozen=True)
class GenerationOptions:
    """Per-call overrides. ``None`` keeps the provider's configured value."""

    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """One completion. ``model`` is the provider's ``name``.

    ``tool_calls`` is only ever non-empty for a ``chat_with_tools`` call; ``metadata``
    then records ``"tool_calling"``: ``"native"`` or ``"emulated"``.

    ``provider_state`` is opaque, provider-owned data that must travel back with this
    reply when the conversation continues -- Anthropic's ``thinking`` blocks (with their
    signatures), Gemini's ``thought_signature`` values. Keyed by provider family
    (``{"anthropic": ...}``), so a provider ignores another's state. :attr:`message`
    carries it in ``Message.metadata["provider_state"]``; append that message (not one
    rebuilt from ``text`` and ``tool_calls``) to keep a tool loop valid on models that
    require it.
    """

    text: str
    usage: TokenUsage | None = None
    model: str = ""
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_state: dict[str, Any] | None = None

    @property
    def message(self) -> Message:
        """This reply as an ``assistant`` message, tool calls included -- what to
        append to the conversation before sending the calls' results back."""

        metadata = {PROVIDER_STATE_KEY: self.provider_state} if self.provider_state else {}
        if not self.tool_calls:
            return Message("assistant", self.text, metadata)
        text = (TextPart(self.text),) if self.text else ()
        return Message(
            "assistant", (*text, *(ToolCallPart(call) for call in self.tool_calls)), metadata
        )


PROVIDER_STATE_KEY = "provider_state"


def provider_state(message: Message, family: str) -> Any:
    """The opaque state provider ``family`` attached to ``message`` (see
    :attr:`LLMResponse.provider_state`), or ``None``."""

    state = message.metadata.get(PROVIDER_STATE_KEY)
    return state.get(family) if isinstance(state, dict) else None


@dataclass(frozen=True)
class StructuredResponse:
    """The result of :meth:`ChatLLMProvider.generate_structured`.

    ``data`` is the parsed JSON value; it is *not* validated against the schema --
    validating is the caller's business (a native schema mode makes violations rare,
    not impossible). ``usage`` sums every call made, repairs included; ``attempts``
    counts them. ``native`` says whether the provider's native mode was used.
    """

    data: Any
    text: str
    usage: TokenUsage | None = None
    model: str = ""
    finish_reason: str | None = None
    native: bool = False
    attempts: int = 1


@runtime_checkable
class LLMProvider(Protocol):
    """Generates a completion from a system + user prompt."""

    def generate(self, system: str, user: str) -> str: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class ChatLLMProvider(LLMProvider, Protocol):
    """The v2 provider surface. See the module docstring.

    ``generate(system, user)`` keeps its v1 meaning and is equivalent to
    ``chat([Message.system(system), Message.user(user)]).text``.
    """

    @property
    def capabilities(self) -> LLMCapabilities: ...

    def chat(
        self, messages: Sequence[Message], *, options: GenerationOptions | None = None
    ) -> LLMResponse: ...

    def generate_structured(
        self,
        messages: Sequence[Message],
        json_schema: dict[str, Any],
        *,
        name: str = "response",
        options: GenerationOptions | None = None,
        max_repair_attempts: int = 1,
    ) -> StructuredResponse: ...


@runtime_checkable
class ToolCallingLLMProvider(ChatLLMProvider, Protocol):
    """The v2 surface plus tool calling.

    A separate protocol rather than new members of :class:`ChatLLMProvider`, so a
    third-party v2 provider written before tool calling still satisfies that one.
    ``tool_choice`` is ``"auto"``, ``"none"``, ``"required"`` or the name of one of
    ``tools``. Every built-in provider, :class:`~kuhaku.core.llm.provider.BaseLLMProvider`
    subclass and :class:`~kuhaku.core.llm.wrapper.LLMWrapper` implements it;
    :func:`~kuhaku.core.llm.wrapper.as_tool_calling_provider` lifts anything else.
    """

    def chat_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        tool_choice: str = "auto",
        options: GenerationOptions | None = None,
    ) -> LLMResponse: ...


def is_retryable_request_exception(exc: BaseException) -> bool:
    """True for connection errors/timeouts, rate limits (429), request timeouts (408),
    and 5xx responses; false for other 4xx (retry won't help a bad request or auth failure)
    and anything else.

    Shared by all three providers (Ollama, Anthropic, OpenAI) -- each currently wraps the
    same ``requests.RequestException`` into ``LLMError`` identically, regardless of status
    code, so this is the one place that distinguishes retryable from non-retryable.
    """

    if isinstance(exc, requests.HTTPError):
        if exc.response is None:
            return False
        return exc.response.status_code in {408, 429} or exc.response.status_code >= 500
    return isinstance(exc, requests.RequestException)


def _status_of(exc: BaseException) -> int | None:
    # requests puts the status on `.response.status_code`; google-genai's APIError
    # carries it as `.code`.
    response = getattr(exc, "response", None)
    for value in (
        getattr(response, "status_code", None),
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, requests.Timeout):
        return True
    # httpx (the google-genai transport) without importing it: TimeoutException and
    # its subclasses all end in "Timeout".
    return type(exc).__module__.startswith("httpx") and "Timeout" in type(exc).__name__


def _is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.ConnectionError):
        return True
    return type(exc).__module__.startswith("httpx") and isinstance(exc, Exception) and (
        "Connect" in type(exc).__name__ or "Network" in type(exc).__name__
    )


def llm_error_from(exc: BaseException, message: str) -> LLMError:
    """The typed :class:`LLMError` for a transport/API exception, with ``message``.

    The caller raises it ``from exc`` so the original stays on ``__cause__``.
    """

    status = _status_of(exc)
    if status == 429:
        from .limits.rate_limit import retry_after_seconds

        return LLMRateLimitError(message, retry_after=retry_after_seconds(exc))
    if status in (401, 403):
        return LLMAuthError(message, status_code=status)
    if status == 408:
        return LLMTimeoutError(message, status_code=status)
    if status is not None and 400 <= status < 500:
        return LLMBadRequestError(message, status_code=status)
    if status is not None and status >= 500:
        return LLMServerError(message, status_code=status)
    if _is_timeout(exc):
        return LLMTimeoutError(message)
    if _is_connection_error(exc):
        return LLMConnectionError(message)
    return LLMError(message)


def counts_as_breaker_failure(exc: BaseException) -> bool:
    """Whether ``exc`` says the provider is unhealthy -- what a circuit breaker counts.

    Everything counts except an HTTP 4xx. A 4xx (400/401/404/413...) is the caller's
    request being wrong, and must not open the circuit for every other caller; a 429 is
    the account's quota, which a rate limiter paces -- opening the circuit on it would
    hide the 429 (and its ``Retry-After``) behind a ``CircuitOpenError``. 408 (request
    timeout) does count: it says the provider is not keeping up.
    """

    status = _status_of(exc)
    return status is None or status == 408 or not 400 <= status < 500
