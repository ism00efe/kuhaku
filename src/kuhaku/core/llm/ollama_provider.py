"""Local LLM via Ollama's HTTP API (the default provider).

Structured output passes the JSON Schema as ``format`` (Ollama 0.5+), which constrains
decoding to the schema. Streaming reads ``/api/chat``'s newline-delimited JSON.

Tool calling sends ``tools`` (``{"type": "function", ...}``); the reply's
``message.tool_calls`` carry arguments as an object and usually no id, so one is
generated. Ollama has no ``tool_choice``: ``"none"`` sends no tools at all, a named
tool is sent alone, and ``"required"``/a named tool add a system instruction to call
it -- which the model may still ignore (``metadata["tool_choice_enforced"]`` is
``False`` then). A stream delivers each call whole, in one message chunk. Whether a
model supports tools at all is the model's business: one that does not makes Ollama
answer 400, surfaced as ``LLMError``; ``native_tool_calling=False`` emulates instead.

Images go in a message's ``images`` field as base64, after the message's text; Ollama
has no PDF input and does not fetch URLs. Only vision models accept images, and which
model is pulled is not knowable offline, so images are declared only for a model whose
name is a known vision family (``llava``, ``llama3.2-vision``, ``qwen2.5vl``, ...);
``content_types=`` overrides that. A user turn may carry images on the tool-calling
path too.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

import requests

from ..media import ImagePart
from ..models import Message, ToolCall
from ..retry import CircuitBreaker, CircuitOpenError
from .base import (
    GenerationOptions,
    LLMCapabilities,
    LLMError,
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
    is_text_only,
    message_text,
    normalize_messages,
    option,
    resolve_content_types,
)
from .structured import StructuredRequest
from .tool_calling import (
    ToolRequest,
    split_tool_message,
    tool_call_from_raw,
    tool_names_by_id,
)

logger = logging.getLogger(__name__)

# Ollama library models that are vision models in every size. Matched against the name
# without its tag or namespace ("llava:13b" -> "llava"). Families where only some sizes
# see images (gemma3:1b does not) are left out on purpose.
_VISION_FAMILIES = frozenset(
    {
        "llava",
        "llava-llama3",
        "llava-phi3",
        "bakllava",
        "llama3.2-vision",
        "llama4",
        "qwen2.5vl",
        "minicpm-v",
        "moondream",
        "granite3.2-vision",
    }
)


class OllamaProvider(ResilientCallMixin, BaseLLMProvider):
    """Calls a local Ollama server's ``/api/chat`` endpoint."""

    capabilities = LLMCapabilities(
        chat=True,
        structured_output=StructuredOutputMode.SCHEMA,
        tool_calling=True,
    )
    _encodable_content_types = frozenset({"text", "image"})

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 180,
        retry_enabled: bool = True,
        retry_max_attempts: int = 3,
        retry_backoff_base_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 10.0,
        circuit_breaker_enabled: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_reset_timeout_seconds: float = 60.0,
        circuit_breaker_success_threshold: int = 1,
        native_tool_calling: bool = True,
        content_types: str | Iterable[str] | None = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise LLMError("OLLAMA_BASE_URL is not set but LLM_PROVIDER=ollama.")
        if not model or not model.strip():
            raise LLMError("OLLAMA_MODEL is not set but LLM_PROVIDER=ollama.")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
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
                name="llm:ollama",
            )
            if circuit_breaker_enabled
            else None
        )
        family = model.strip().lower().split(":", 1)[0].rsplit("/", 1)[-1]
        self.capabilities = replace(
            type(self).capabilities,
            content_types=resolve_content_types(
                content_types,
                default=(
                    self._encodable_content_types
                    if family in _VISION_FAMILIES
                    else frozenset({"text"})
                ),
                encodable=self._encodable_content_types,
                provider=self.name,
            ),
        )
        if not native_tool_calling:
            self.capabilities = replace(self.capabilities, tool_calling=False)
        # Set by every call when Ollama reports usage; read by TokenTrackingLLM.generate.
        self.last_usage: TokenUsage | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

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
        payload["format"] = request.schema
        return self._complete(payload)

    def _tool_complete(
        self,
        messages: tuple[Message, ...],
        request: ToolRequest,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        response = self._complete(self._payload(messages, options, tools=request))
        if request.choice in ("auto", "none"):
            return response
        return replace(response, metadata={**response.metadata, "tool_choice_enforced": False})

    # --- transport --------------------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[Message],
        options: GenerationOptions | None,
        *,
        tools: ToolRequest | None = None,
    ) -> dict[str, Any]:
        options = options or GenerationOptions()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._message(m) for m in messages]
            if tools is None
            else self._tool_messages(messages, tools),
            "stream": False,
            "options": {
                "temperature": option(options.temperature, self._temperature),
                "num_predict": option(options.max_tokens, self._max_tokens),
            },
        }
        if tools is not None and tools.choice != "none":
            specs = [s for s in tools.tools if s.name == tools.forced] if tools.forced else tools.tools
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in specs
            ]
        return payload

    def _tool_messages(
        self, messages: Sequence[Message], tools: ToolRequest
    ) -> list[dict[str, Any]]:
        names = tool_names_by_id(messages)
        encoded: list[dict[str, Any]] = []
        for message in messages:
            split = split_tool_message(message, provider=self.name)
            if split.results:
                for result in split.results:
                    item = {"role": "tool", "content": result.content}
                    name = result.name or names.get(result.tool_call_id)
                    if name:
                        item["tool_name"] = name
                    encoded.append(item)
            elif split.calls:
                encoded.append(
                    {
                        "role": "assistant",
                        "content": split.text,
                        "tool_calls": [
                            {"function": {"name": call.name, "arguments": call.arguments}}
                            for call in split.calls
                        ],
                    }
                )
            elif split.has_media:
                encoded.append(self._message(message))
            else:
                encoded.append({"role": message.role, "content": split.text})
        if tools.choice == "required" or tools.forced:
            rule = (
                f'Call the tool "{tools.forced}" now.'
                if tools.forced
                else "Call one of the available tools now."
            )
            # After the leading system turns, as the structured-output instruction goes.
            split_at = next(
                (i for i, item in enumerate(encoded) if item["role"] != "system"), len(encoded)
            )
            encoded.insert(split_at, {"role": "system", "content": rule})
        return encoded

    def _message(self, message: Message) -> dict[str, Any]:
        if is_text_only(message):
            return {"role": message.role, "content": message_text(message, provider=self.name)}
        check_media(message, provider=self.name, content_types=self.capabilities.content_types)
        images: list[str] = []
        for part in message.parts:
            if isinstance(part, ImagePart) and part.data is not None:
                images.append(part.base64)
            elif isinstance(part, ImagePart):
                raise UnsupportedContentError(
                    f"{self.name} cannot fetch an image URL; pass the image's bytes "
                    f"(ImagePart.from_bytes or .from_path)."
                )
        return {"role": message.role, "content": message.text, "images": images}

    def _post(self, payload: dict[str, Any], *, stream: bool = False) -> requests.Response:
        kwargs: dict[str, Any] = {"stream": True} if stream else {}
        resp = requests.post(
            f"{self._base_url}/api/chat",
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
            raise llm_error_from(
                exc,
                f"Failed to reach Ollama at '{self._base_url}' for model '{self._model}'.\n"
                f"Original error: {exc}\n"
                f"Possible solutions:\n"
                f"1. Start Ollama and pull the model:\n"
                f"   ollama serve\n"
                f"   ollama pull {self._model}\n"
                f"2. Or switch to a hosted provider by setting:\n"
                f"   KUHAKU_LLM_PROVIDER=openai and OPENAI_API_KEY=<key>"
            ) from exc

    def _complete(self, payload: dict[str, Any]) -> LLMResponse:
        data = self._send(lambda: self._post(payload).json())
        try:
            content = data["message"]["content"].strip()
            tool_calls = _tool_calls(data["message"])
        except (KeyError, TypeError, AttributeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {data}") from exc

        # Best-effort token usage logging (Ollama includes it in the non-streaming
        # response). Never changes the return value.
        usage = _usage(data)
        if usage is not None:
            self.last_usage = usage
            logger.info(
                "llm usage",
                extra={
                    "status": "ok",
                    "token_count": usage.completion_tokens,
                    "prompt_token_count": usage.prompt_tokens,
                },
            )
        return LLMResponse(
            text=content,
            usage=usage,
            model=self.name,
            finish_reason=data.get("done_reason"),
            tool_calls=tool_calls,
        )


def _tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    return tuple(
        tool_call_from_raw(
            call["function"]["name"], call["function"].get("arguments"), id=call.get("id")
        )
        for call in message.get("tool_calls") or []
    )


def _usage(data: dict[str, Any]) -> TokenUsage | None:
    completion = data.get("eval_count")
    prompt = data.get("prompt_eval_count")
    if completion is None and prompt is None:
        return None
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion)
