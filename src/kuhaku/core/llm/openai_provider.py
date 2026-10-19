"""OpenAI (or any OpenAI-compatible endpoint) via the Chat Completions HTTP API.

Uses ``requests`` for symmetry with the other providers. Because it targets the standard
``/v1/chat/completions`` shape, it also works with compatible servers (vLLM, LM Studio,
etc.) by changing ``OPENAI_BASE_URL``. Selected with ``LLM_PROVIDER=openai``.

Structured output uses ``response_format={"type": "json_schema", ...}``. Not every
compatible server accepts that; pass ``native_structured_output=False`` for one that
does not, and structured output falls back to prompt + parse.

Tool calling sends ``tools`` (``{"type": "function", ...}``) and ``tool_choice``; the
reply's ``message.tool_calls`` carry their arguments as JSON text, parsed here. A
stream sends each call in fragments keyed by ``index``, assembled before the final
chunk. ``native_tool_calling=False`` emulates tool calling for a compatible server
without it.

Reasoning models (``gpt-5*``, ``o1*``, ``o3*``, ``o4*``) take ``max_completion_tokens``
instead of ``max_tokens`` (which they reject) and reject a non-default ``temperature``,
so for them the limit is sent under the new name and the temperature is dropped (with a
one-time warning). Every other model's payload is unchanged.

A reply with no content and no tool calls -- a refusal, or a reasoning model that spent
its whole budget thinking (``finish_reason: "length"``) -- raises a clear
:class:`~kuhaku.core.llm.base.LLMError` (:class:`~kuhaku.core.llm.base.LLMRefusalError`
for a refusal) instead of a parse error.

Images go as ``image_url`` parts (a ``data:`` URL for inline bytes), PDFs as ``file``
parts with base64 ``file_data``. Which of them an instance declares in
``capabilities.content_types`` depends on the model -- see
:meth:`OpenAIProvider._default_content_types` -- and can be set with ``content_types=``. A user turn may carry them on the tool-calling
path too.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

import requests

from ..media import DocumentPart, ImagePart
from ..models import ContentPart, Message, TextPart
from ..retry import CircuitBreaker, CircuitOpenError
from .base import (
    GenerationOptions,
    LLMCapabilities,
    LLMError,
    LLMRefusalError,
    LLMResponse,
    StructuredOutputMode,
    TokenUsage,
    UnsupportedContentError,
    llm_error_from,
)
from .provider import (
    BaseLLMProvider,
    ResilientCallMixin,
    check_media,
    http_url,
    is_text_only,
    message_text,
    normalize_messages,
    option,
    resolve_content_types,
)
from .structured import StructuredRequest
from .tool_calling import (
    ToolRequest,
    encoded_arguments,
    split_tool_message,
    tool_call_from_raw,
)

logger = logging.getLogger(__name__)

_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Model families OpenAI documents as accepting image and PDF input. A prefix match, so
# dated snapshots (gpt-4o-2024-08-06) count; the markers exclude same-prefix models
# that take other modalities instead.
_MULTIMODAL_PREFIXES = ("gpt-4o", "gpt-4.1", "gpt-5", "o3", "o4")
_NOT_MULTIMODAL = ("audio", "realtime", "transcribe", "tts", "search", "o3-mini")

# OpenAI's reasoning families: max_completion_tokens only, default temperature only.
# Matched on the bare model id; a provider-prefixed id behind a gateway
# ("openai/gpt-oss-120b" on Groq) is a different API and keeps max_tokens.
_REASONING_MODEL = re.compile(r"^(?:gpt-5|o1|o3|o4)(?:$|[-.])")


def is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL.match(model.strip().lower()))


class OpenAIProvider(ResilientCallMixin, BaseLLMProvider):
    capabilities = LLMCapabilities(
        chat=True,
        structured_output=StructuredOutputMode.SCHEMA,
        tool_calling=True,
    )
    _error_prefix = "OpenAI"
    # OpenAI only reports usage on a stream when asked to.
    _stream_usage_option = True
    _encodable_content_types = frozenset({"text", "image", "document"})

    # Subclasses for other OpenAI-compatible services set their own, so the circuit
    # breaker's messages name the service that is actually down.
    _service = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120,
        retry_enabled: bool = True,
        retry_max_attempts: int = 3,
        retry_backoff_base_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 10.0,
        circuit_breaker_enabled: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_reset_timeout_seconds: float = 60.0,
        circuit_breaker_success_threshold: int = 1,
        native_structured_output: bool = True,
        native_tool_calling: bool = True,
        content_types: str | Iterable[str] | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai.")
        if not model or not model.strip():
            raise LLMError("OPENAI_MODEL is not set but LLM_PROVIDER=openai.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning = is_reasoning_model(model)
        self._warned_sampling = False
        self._timeout = timeout
        self._retry_enabled = retry_enabled
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_base_seconds = retry_backoff_base_seconds
        self._retry_backoff_max_seconds = retry_backoff_max_seconds
        self._circuit_breaker = (
            CircuitBreaker(
                failure_threshold=circuit_breaker_failure_threshold,
                reset_timeout_seconds=circuit_breaker_reset_timeout_seconds,
                success_threshold=circuit_breaker_success_threshold,
                name=f"llm:{self._service}",
            )
            if circuit_breaker_enabled
            else None
        )
        if not native_structured_output:
            self.capabilities = replace(
                self.capabilities, structured_output=StructuredOutputMode.NONE
            )
        self.capabilities = replace(
            self.capabilities,
            content_types=resolve_content_types(
                content_types,
                default=self._default_content_types(),
                encodable=self._encodable_content_types,
                provider=self.name,
            ),
        )
        if not native_tool_calling:
            self.capabilities = replace(self.capabilities, tool_calling=False)
        # Set by every call when OpenAI reports usage; read by TokenTrackingLLM.generate.
        self.last_usage: TokenUsage | None = None

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def _default_content_types(self) -> frozenset[str]:
        """Text, plus image and document for the model families OpenAI documents as
        multimodal -- but only against OpenAI itself: behind another base URL (vLLM, a
        gateway) the model is unknown, so nothing beyond text is claimed."""

        model = self._model.lower()
        if (
            self._base_url == _OPENAI_BASE_URL
            and model.startswith(_MULTIMODAL_PREFIXES)
            and not any(marker in model for marker in _NOT_MULTIMODAL)
        ):
            return frozenset({"text", "image", "document"})
        return frozenset({"text"})

    # --- v2 hooks ---------------------------------------------------------------------

    def chat(
        self, messages: Sequence[Message], *, options: GenerationOptions | None = None
    ) -> LLMResponse:
        return self._complete(self._payload(normalize_messages(messages), options))

    def _structured_complete(
        self,
        messages: tuple[Message, ...],
        request: StructuredRequest,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        payload = self._payload(messages, options)
        payload["response_format"] = self._response_format(request)
        return self._complete(payload)

    def _response_format(self, request: StructuredRequest) -> dict[str, Any]:
        # strict=False: strict mode rejects schemas that do not mark every property
        # required and forbid additionalProperties, which ordinary Pydantic schemas do
        # not. Non-strict still constrains decoding to the schema on current models.
        return {
            "type": "json_schema",
            "json_schema": {"name": request.name, "schema": request.schema, "strict": False},
        }

    def _tool_complete(
        self,
        messages: tuple[Message, ...],
        request: ToolRequest,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        return self._complete(self._payload(messages, options, tools=request))

    # --- transport --------------------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[Message],
        options: GenerationOptions | None,
        *,
        tools: ToolRequest | None = None,
    ) -> dict[str, Any]:
        options = options or GenerationOptions()
        payload: dict[str, Any] = {"model": self._model}
        if self._reasoning:
            # Debug, not warning: llm_temperature always has a value (its default), so a
            # warning would fire for every reasoning-model user who never set one.
            if not self._warned_sampling:
                self._warned_sampling = True
                logger.debug(
                    "%s is a reasoning model and takes only its default temperature; "
                    "the configured temperature is not sent.",
                    self.name,
                )
            payload["max_completion_tokens"] = option(options.max_tokens, self._max_tokens)
        else:
            payload["temperature"] = option(options.temperature, self._temperature)
            payload["max_tokens"] = option(options.max_tokens, self._max_tokens)
        payload |= {
            "messages": [
                self._tool_message(m) if tools is not None else self._message(m)
                for m in messages
            ],
        }
        if tools is not None:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in tools.tools
            ]
            payload["tool_choice"] = (
                {"type": "function", "function": {"name": tools.forced}}
                if tools.forced
                else tools.choice
            )
        return payload

    def _tool_message(self, message: Message) -> dict[str, Any]:
        split = split_tool_message(message, provider=self.name)
        if split.results:
            if len(split.results) != 1:
                raise UnsupportedContentError(
                    f"{self.name}: send one tool result per tool message."
                )
            result = split.results[0]
            return {"role": "tool", "tool_call_id": result.tool_call_id, "content": result.content}
        if split.calls:
            return {
                "role": "assistant",
                # The API takes null, not "", for an assistant turn that only calls tools.
                "content": split.text or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": encoded_arguments(call)},
                    }
                    for call in split.calls
                ],
            }
        if split.has_media:
            return self._message(message)
        return {"role": message.role, "content": split.text}

    def _message(self, message: Message) -> dict[str, Any]:
        if is_text_only(message):
            return {"role": message.role, "content": message_text(message, provider=self.name)}
        check_media(message, provider=self.name, content_types=self.capabilities.content_types)
        return {"role": message.role, "content": [self._part(part) for part in message.parts]}

    def _part(self, part: ContentPart) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {"type": "text", "text": part.text}
        if isinstance(part, ImagePart):
            url = part.data_url if part.url is None else http_url(part, provider=self.name)
            image: dict[str, Any] = {"url": url}
            if part.detail is not None:
                image["detail"] = part.detail
            return {"type": "image_url", "image_url": image}
        if isinstance(part, DocumentPart) and part.data is not None:
            return {
                "type": "file",
                "file": {"filename": part.filename or "document.pdf", "file_data": part.data_url},
            }
        raise UnsupportedContentError(
            f"{self.name} cannot send a {part.type} by URL; pass its bytes "
            f"({type(part).__name__}.from_bytes or .from_path)."
        )

    def _post(self, payload: dict[str, Any], *, stream: bool = False) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        kwargs: dict[str, Any] = {"stream": True} if stream else {}
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout,
            **kwargs,
        )
        resp.raise_for_status()
        return resp

    def _send(self, fn: Any) -> Any:
        try:
            return self._resilient(fn)
        except CircuitOpenError as exc:
            raise LLMError(str(exc)) from exc
        except requests.RequestException as exc:
            raise llm_error_from(exc, f"{self._error_prefix} request failed: {exc}") from exc

    def _complete(self, payload: dict[str, Any]) -> LLMResponse:
        data = self._send(lambda: self._post(payload).json())
        try:
            choice = data["choices"][0]
            raw_calls = choice["message"].get("tool_calls") or []
            tool_calls = tuple(
                tool_call_from_raw(
                    call["function"]["name"],
                    call["function"].get("arguments"),
                    id=call.get("id"),
                )
                for call in raw_calls
            )
            content = choice["message"].get("content")
            refusal = choice["message"].get("refusal")
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(
                f"Unexpected {self._error_prefix} response shape: {type(exc).__name__}: {exc}"
            ) from exc
        if refusal and not tool_calls:
            raise LLMRefusalError(f"{self.name} refused: {refusal}")
        if not isinstance(content, str):
            if not tool_calls:
                if finish_reason == "length":
                    raise LLMError(
                        f"{self.name} produced no output before reaching the token limit "
                        f"(finish_reason=length); raise max_tokens -- a reasoning model "
                        f"spends part of it thinking."
                    )
                raise LLMError(
                    f"{self.name} returned no content and no tool calls "
                    f"(finish_reason={finish_reason})."
                )
            # A reply that only calls tools has null content.
            content = ""
        content = content.strip()

        # Best-effort token usage logging. Never changes the return type.
        usage = data.get("usage") or {}
        token_usage = _usage(usage)
        if token_usage is not None:
            self.last_usage = token_usage
            logger.info(
                "llm usage",
                extra={
                    "status": "ok",
                    "token_count": usage.get("completion_tokens"),
                    "prompt_token_count": usage.get("prompt_tokens"),
                },
            )
        return LLMResponse(
            text=content,
            usage=token_usage,
            model=self.name,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )


def _usage(usage: dict[str, Any] | None) -> TokenUsage | None:
    if not usage:
        return None
    return TokenUsage(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )
