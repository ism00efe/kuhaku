"""Google Vertex AI (Gemini) via the ``google-genai`` SDK.

Optional provider for users with Google Cloud credits who want to avoid running a local
model. The SDK is imported lazily so ``google-genai`` stays an optional dependency (see
``pyproject.toml``'s ``vertex`` extra) -- importing this module without it installed does
not fail; only instantiating :class:`VertexAIProvider` does. Selected with
``LLM_PROVIDER=vertex``.

Structured output sets ``response_mime_type="application/json"`` plus the schema
(``response_json_schema``, or ``response_schema`` on older SDKs); streaming uses
``generate_content_stream``.

Tool calling sends ``tools=[{"function_declarations": [...]}]`` (parameters as
``parameters_json_schema``, or ``parameters`` on older SDKs) and ``tool_config``'s
``function_calling_config`` (``AUTO``/``ANY``/``NONE``, ``allowed_function_names`` for a
named tool). The reply's ``function_call`` parts become tool calls; results go back as
``function_response`` parts on a user turn, named after the call -- a result without a
name is matched to its call by id within the conversation. Call ids are generated
locally and never sent: the API pairs calls and responses by order and name. A stream
delivers each call whole, in the chunk that carries it.

Gemini 3 attaches a ``thought_signature`` to its ``function_call`` parts (and may to
others) and rejects a later turn that does not send it back. The reply's raw parts are
kept as the response's ``provider_state`` whenever one carries a signature, and that
assistant turn is re-sent from them verbatim.

``temperature``/``max_tokens`` given to the constructor are the defaults for every call
(per-call options override them); ``None`` leaves the model's own default -- which for
Gemini 3 is the recommended temperature. Usage counts thinking tokens
(``thoughts_token_count``, billed as output) in ``completion_tokens``.

Images and PDFs become ``inline_data`` parts (the raw bytes; the SDK encodes them) or,
for a URL (``gs://`` included), ``file_data`` parts that Vertex fetches itself. Every
Gemini model accepts both, so a ``gemini-*`` model declares them by default;
``content_types=`` overrides that. A user turn may carry them on the tool-calling path
too.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

from ..media import MediaPart
from ..models import Message, TextPart, ToolCall
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
    provider_state,
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
    split_system,
)
from .structured import StructuredRequest
from .tool_calling import (
    ToolRequest,
    ToolSpec,
    split_tool_message,
    tool_call_from_raw,
    tool_names_by_id,
)

logger = logging.getLogger(__name__)

_MISSING_CONFIG_MESSAGE = """Vertex AI configuration is missing. Please set `vertex_project` and `vertex_location` via Settings:

from kuhaku.core.config import get_settings
get_settings().vertex_project = "your-project-id"
get_settings().vertex_location = "us-central1\""""


def is_retryable_vertex_exception(exc: BaseException) -> bool:
    """True for connection errors/timeouts, rate limits (429), request timeouts (408),
    and 5xx API errors; false for other 4xx (retry won't help a bad request or auth failure)
    and anything else -- mirrors ``is_retryable_request_exception`` for the google-genai
    SDK's own exception types, which are not ``requests.RequestException`` subclasses.
    """

    import httpx
    import requests
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        return exc.code in {408, 429} or exc.code >= 500
    return isinstance(exc, (httpx.RequestError, requests.RequestException))


class VertexAIProvider(ResilientCallMixin, BaseLLMProvider):
    capabilities = LLMCapabilities(
        chat=True,
        structured_output=StructuredOutputMode.SCHEMA,
        tool_calling=True,
    )
    _encodable_content_types = frozenset({"text", "image", "document"})

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        project: str | None = None,
        location: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 120,
        retry_enabled: bool = True,
        retry_max_attempts: int = 3,
        retry_backoff_base_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 10.0,
        circuit_breaker_enabled: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_reset_timeout_seconds: float = 60.0,
        circuit_breaker_success_threshold: int = 1,
        content_types: str | Iterable[str] | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Vertex AI support requires: pip install google-genai"
            ) from exc

        self._model = model
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        if not self._project or not self._location:
            raise ValueError(_MISSING_CONFIG_MESSAGE)
        try:
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        except Exception as exc:
            raise ValueError(_MISSING_CONFIG_MESSAGE) from exc

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
                name="llm:vertex",
            )
            if circuit_breaker_enabled
            else None
        )
        self.capabilities = replace(
            type(self).capabilities,
            content_types=resolve_content_types(
                content_types,
                default=(
                    self._encodable_content_types
                    if model.lower().startswith("gemini-")
                    else frozenset({"text"})
                ),
                encodable=self._encodable_content_types,
                provider=self.name,
            ),
        )
        # Set by generate() when Vertex reports usage; read by TokenTrackingLLM.
        self.last_usage: TokenUsage | None = None

    @property
    def name(self) -> str:
        return f"vertex:{self._model}"

    # --- v2 hooks ---------------------------------------------------------------------

    def chat(
        self, messages: Sequence[Message], *, options: GenerationOptions | None = None
    ) -> LLMResponse:
        contents, config = self._request(normalize_messages(messages), options)
        return self._complete(contents, config)

    def _structured_complete(
        self,
        messages: tuple[Message, ...],
        request: StructuredRequest,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        from google.genai import types

        contents, config = self._request(messages, options)
        config["response_mime_type"] = "application/json"
        # response_json_schema takes full JSON Schema ($defs/$ref included, as Pydantic
        # emits); SDKs older than it only have response_schema, an OpenAPI subset.
        if "response_json_schema" in getattr(types.GenerateContentConfig, "model_fields", {}):
            config["response_json_schema"] = request.schema
        else:
            config["response_schema"] = request.schema
        return self._complete(contents, config)

    def _tool_complete(
        self,
        messages: tuple[Message, ...],
        request: ToolRequest,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        contents, config = self._request(messages, options, tools=request)
        return self._complete(contents, config)

    # --- transport --------------------------------------------------------------------

    def _request(
        self,
        messages: Sequence[Message],
        options: GenerationOptions | None,
        *,
        tools: ToolRequest | None = None,
    ) -> tuple[Any, Any]:
        system, rest = split_system(messages, provider=self.name)
        if tools is not None:
            contents: Any = self._tool_contents(rest)
        elif len(rest) == 1 and rest[0].role == "user" and is_text_only(rest[0]):
            # The SDK's shorthand for one user turn -- and exactly what generate() has
            # always sent.
            contents = message_text(rest[0], provider=self.name)
        else:
            contents = [
                {"role": "model" if m.role == "assistant" else "user", "parts": self._parts(m)}
                for m in rest
            ]
            for entry, message in zip(contents, rest, strict=True):
                replayed = _replayed_parts(message)
                if replayed is not None:
                    entry["parts"] = replayed
        config: Any = {}
        if system is not None:
            config["system_instruction"] = system
        # SDK timeout is in milliseconds.
        config["http_options"] = {"timeout": self._timeout * 1000}
        # Only what was configured or passed is sent; otherwise the model's default holds.
        options = options or GenerationOptions()
        temperature = option(options.temperature, self._temperature)
        max_tokens = option(options.max_tokens, self._max_tokens)
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if tools is not None:
            config["tools"] = [{"function_declarations": [_declaration(s) for s in tools.tools]}]
            calling: dict[str, Any] = {
                "mode": {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(
                    tools.choice, "ANY"
                )
            }
            if tools.forced:
                calling["allowed_function_names"] = [tools.forced]
            config["tool_config"] = {"function_calling_config": calling}
        return contents, config

    def _tool_contents(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        names = tool_names_by_id(messages)
        contents: list[dict[str, Any]] = []
        for message in messages:
            split = split_tool_message(message, provider=self.name)
            replayed = _replayed_parts(message)
            if replayed is not None:
                contents.append({"role": "model", "parts": replayed})
            elif split.results:
                parts = []
                for result in split.results:
                    name = result.name or names.get(result.tool_call_id)
                    if not name:
                        raise UnsupportedContentError(
                            f"{self.name}: the tool result for call "
                            f"'{result.tool_call_id}' has no name, and no call with that "
                            f"id is in the conversation."
                        )
                    key = "error" if result.is_error else "result"
                    parts.append(
                        {"function_response": {"name": name, "response": {key: result.content}}}
                    )
                previous = contents[-1] if contents else None
                if previous is not None and "function_response" in previous["parts"][0]:
                    previous["parts"].extend(parts)
                else:
                    contents.append({"role": "user", "parts": parts})
            elif split.calls:
                text = [{"text": split.text}] if split.text else []
                contents.append(
                    {
                        "role": "model",
                        "parts": text
                        + [
                            {"function_call": {"name": call.name, "args": call.arguments}}
                            for call in split.calls
                        ],
                    }
                )
            else:
                contents.append(
                    {
                        "role": "model" if message.role == "assistant" else "user",
                        "parts": self._parts(message) if split.has_media else [{"text": split.text}],
                    }
                )
        return contents

    def _parts(self, message: Message) -> list[dict[str, Any]]:
        if is_text_only(message):
            return [{"text": message_text(message, provider=self.name)}]
        check_media(message, provider=self.name, content_types=self.capabilities.content_types)
        parts: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, TextPart):
                parts.append({"text": part.text})
            elif isinstance(part, MediaPart) and part.data is not None:
                parts.append({"inline_data": {"mime_type": part.media_type, "data": part.data}})
            elif isinstance(part, MediaPart):
                parts.append({"file_data": {"file_uri": part.url, "mime_type": part.media_type}})
            else:  # check_media lets only text and media parts through
                raise UnsupportedContentError(f"{self.name} cannot send {part.type} content.")
        return parts

    def _send(self, fn: Any) -> Any:
        try:
            return self._resilient(fn, retryable=is_retryable_vertex_exception)
        except CircuitOpenError as exc:
            raise LLMError(str(exc)) from exc
        except Exception as exc:
            raise llm_error_from(exc, f"Vertex AI request failed: {exc}") from exc

    def _complete(self, contents: Any, config: Any) -> LLMResponse:
        response = self._send(
            lambda: self._client.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        )

        text = getattr(response, "text", None)
        try:
            tool_calls = _tool_calls(response)
        except (AttributeError, TypeError) as exc:
            raise LLMError(f"Unexpected Vertex AI response: {response!r}") from exc
        if not text and not tool_calls:
            raise LLMError(f"Unexpected Vertex AI response: {response!r}")

        usage = _usage(response)
        if usage is not None:
            self.last_usage = usage

        return LLMResponse(
            text=(text or "").strip(),
            usage=usage,
            model=self.name,
            finish_reason=_finish_reason(response),
            tool_calls=tool_calls,
            provider_state=_state(_raw_parts(response)),
        )


def _declaration(spec: ToolSpec) -> dict[str, Any]:
    from google.genai import types

    # parameters_json_schema takes full JSON Schema, as response_json_schema does for
    # structured output; older SDKs only have parameters, an OpenAPI subset.
    key = (
        "parameters_json_schema"
        if "parameters_json_schema" in getattr(types.FunctionDeclaration, "model_fields", {})
        else "parameters"
    )
    return {"name": spec.name, "description": spec.description, key: spec.parameters}


def _tool_calls(response: Any) -> tuple[ToolCall, ...]:
    calls = getattr(response, "function_calls", None) or []
    return tuple(
        tool_call_from_raw(
            call.name,
            dict(call.args) if call.args is not None else None,
            id=getattr(call, "id", None),
        )
        for call in calls
    )


def _usage(response: Any) -> TokenUsage | None:
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        return None
    candidates = getattr(usage_metadata, "candidates_token_count", None)
    thoughts = getattr(usage_metadata, "thoughts_token_count", None)
    # Thinking tokens are billed as output but reported apart from the candidates.
    completion = (
        None
        if candidates is None and not isinstance(thoughts, int)
        else (candidates or 0) + (thoughts if isinstance(thoughts, int) else 0)
    )
    return TokenUsage(
        prompt_tokens=getattr(usage_metadata, "prompt_token_count", None),
        completion_tokens=completion,
    )


_STATE_KEY = "vertex"


def _raw_parts(response: Any) -> list[dict[str, Any]]:
    """The first candidate's content parts as dicts the SDK accepts back."""

    candidates = getattr(response, "candidates", None) or []
    content = getattr(candidates[0], "content", None) if candidates else None
    out: list[dict[str, Any]] = []
    for part in getattr(content, "parts", None) or []:
        if isinstance(part, dict):
            out.append(dict(part))
        elif callable(getattr(part, "model_dump", None)):
            out.append(part.model_dump(exclude_none=True))
    return out


def _state(parts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not any(part.get("thought_signature") for part in parts):
        return None
    return {_STATE_KEY: {"parts": parts}}


def _replayed_parts(message: Message) -> list[dict[str, Any]] | None:
    if message.role != "assistant":
        return None
    state = provider_state(message, _STATE_KEY)
    parts = state.get("parts") if isinstance(state, dict) else None
    return [dict(part) for part in parts] if isinstance(parts, list) and parts else None


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    if reason is None:
        return None
    return str(getattr(reason, "value", reason))
