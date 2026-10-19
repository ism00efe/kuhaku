"""Anthropic Claude via the Messages HTTP API.

Uses ``requests`` (not the vendor SDK) to keep dependencies minimal and the three
providers symmetric. Selected with ``LLM_PROVIDER=anthropic``.

Request parameters follow what the model accepts (see :func:`model_traits`):

- ``temperature`` is sent only when it was set explicitly *and* the model takes
  sampling parameters. Current Claude models (Opus 4.7 and later, Sonnet 5 and later,
  Haiku 5.5, Fable/Mythos) reject a non-default ``temperature`` with a 400, so on them
  an explicit value is dropped with a warning. Opus/Sonnet 4.6 and older, Haiku 4.5,
  and any model id that is not ``claude-*`` (a proxy alias) still receive it.
- ``max_tokens`` defaults to 16000 on current Claude models, where adaptive thinking is
  on by default and its tokens count against ``max_tokens`` -- 1024 would often be
  spent on thinking before any answer. Older and unknown models keep 1024.

Structured output on current Claude models uses native structured outputs
(``output_config.format`` with a ``json_schema``): the reply text is the JSON. On older
and unknown models it is a forced tool call: the schema becomes a tool's
``input_schema`` and ``tool_choice`` names that tool. Either way a schema whose root is
not an object is wrapped in one (``{"value": ...}``, ``$defs`` hoisted to the new root)
and unwrapped again.

Tool calling sends ``tools`` (``input_schema``) and ``tool_choice``. ``auto``/``none``
map directly; ``required`` and a named tool map to ``any``/``tool`` on models that
accept forced tool use. Claude Opus 5.5, Sonnet 5.5, Fable 5.1 and Mythos 5.1 (and,
conservatively, any newer ``claude-*`` id) reject forced tool use with a 400, so there
``auto`` is sent with a system instruction to call the tool -- a strong steer, not a
guarantee. The reply's ``tool_use`` blocks become tool calls. Results go back as
``tool_result`` blocks on a user turn -- consecutive ``tool`` messages are merged into
one, since the API requires turns to alternate. A stream assembles each ``tool_use``
block from its ``input_json_delta`` fragments.

``thinking``/``redacted_thinking`` blocks (with their signatures) are kept as the
response's ``provider_state`` and sent back verbatim, with the rest of that assistant
turn, when the conversation continues: models with thinking reject a tool loop whose
earlier assistant turns lost them.

Images and PDFs are ``image`` / ``document`` content blocks, inline (base64) or by URL.
Every current Claude model accepts both, so a ``claude-*`` model declares them by
default; ``content_types=`` overrides that. A user turn may carry them on the
tool-calling path too.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import requests

from ..media import DocumentPart, ImagePart, MediaPart
from ..models import ContentPart, Message, TextPart, ToolCall
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
    provider_state,
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
    split_system,
)
from .structured import StructuredRequest, wrap_in_object
from .tool_calling import ToolRequest, split_tool_message, tool_call_from_raw

logger = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_WRAPPED_KEY = "value"
_STATE_KEY = "anthropic"
_THINKING_TYPES = frozenset({"thinking", "redacted_thinking"})

DEFAULT_MAX_TOKENS = 16000
LEGACY_MAX_TOKENS = 1024

# Models that still take temperature/top_p/top_k: Claude 3.x, Opus/Sonnet 4.0, 4.1,
# 4.5, 4.6 and Haiku 4.5 (aliases and dated snapshots). Opus 4.7+, Sonnet 5+, Haiku
# 5.5 and Fable/Mythos reject non-default sampling parameters.
_LEGACY = re.compile(r"^claude-(?:3|instant|(?:opus|sonnet|haiku)-4(?:-[0156])?(?:$|@|-\d{8}))")
# Models known to accept forced tool use (tool_choice any/tool). Opus 5.5, Sonnet 5.5,
# Fable 5.1 and Mythos 5.1 return a 400; a claude-* id not listed here is assumed to
# behave like the newest models.
_FORCED_TOOL_OK = re.compile(
    r"^claude-(?:3|instant|(?:opus|sonnet|haiku)-4|(?:opus|sonnet|fable|mythos)-5(?:$|@|-\d{8})"
    r"|haiku-5-5)"
)


@dataclass(frozen=True)
class ModelTraits:
    """What the Messages API accepts for one model id."""

    sampling: bool
    """``temperature`` may be sent."""

    forced_tool_choice: bool
    """``tool_choice`` ``any``/``tool`` may be sent."""

    native_structured_output: bool
    """``output_config.format`` is used for structured output."""

    default_max_tokens: int


def model_traits(model: str) -> ModelTraits:
    """The request traits for ``model``. A non-``claude-*`` id (a proxy alias, a test
    name) gets the conservative pre-2026 behaviour: everything is sent as before."""

    name = model.strip().lower()
    if not name.startswith("claude-"):
        return ModelTraits(True, True, False, LEGACY_MAX_TOKENS)
    if _LEGACY.match(name):
        return ModelTraits(True, True, False, LEGACY_MAX_TOKENS)
    return ModelTraits(
        sampling=False,
        forced_tool_choice=bool(_FORCED_TOOL_OK.match(name)),
        native_structured_output=True,
        default_max_tokens=DEFAULT_MAX_TOKENS,
    )


class AnthropicProvider(ResilientCallMixin, BaseLLMProvider):
    capabilities = LLMCapabilities(
        chat=True,
        structured_output=StructuredOutputMode.SCHEMA,
        tool_calling=True,
    )
    _encodable_content_types = frozenset({"text", "image", "document"})

    def __init__(
        self,
        api_key: str,
        model: str,
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
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic.")
        if not model or not model.strip():
            raise LLMError("ANTHROPIC_MODEL is not set but LLM_PROVIDER=anthropic.")
        self._api_key = api_key
        self._model = model
        self._traits = model_traits(model)
        # None: not configured -- the API's own default applies (and nothing is sent).
        self._temperature = temperature
        self._max_tokens = self._traits.default_max_tokens if max_tokens is None else max_tokens
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
                name="llm:anthropic",
            )
            if circuit_breaker_enabled
            else None
        )
        default = (
            self._encodable_content_types
            if model.lower().startswith("claude-")
            else frozenset({"text"})
        )
        self.capabilities = replace(
            type(self).capabilities,
            content_types=resolve_content_types(
                content_types,
                default=default,
                encodable=self._encodable_content_types,
                provider=self.name,
            ),
        )
        # Set by every call when Anthropic reports usage; read by TokenTrackingLLM.generate.
        self.last_usage: TokenUsage | None = None

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

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
        schema = request.schema
        wrapped = schema.get("type") != "object"
        root = wrap_in_object(schema, _WRAPPED_KEY) if wrapped else schema
        payload = self._payload(messages, options)
        if self._traits.native_structured_output:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": strict_output_schema(root)}
            }
            return self._complete(payload, structured=True, unwrap=wrapped)
        payload["tools"] = [
            {
                "name": request.name,
                "description": f"Record the {request.name} as structured data.",
                "input_schema": root,
            }
        ]
        payload["tool_choice"] = {"type": "tool", "name": request.name}
        return self._complete(payload, tool=request.name, unwrap=wrapped)

    def _tool_complete(
        self,
        messages: tuple[Message, ...],
        request: ToolRequest,
        options: GenerationOptions | None,
    ) -> LLMResponse:
        return self._complete(self._payload(messages, options, tools=request))

    # --- transport --------------------------------------------------------------------

    def _sampling(self, options: GenerationOptions) -> dict[str, Any]:
        temperature = option(options.temperature, self._temperature)
        if temperature is None:
            return {}
        if self._traits.sampling:
            return {"temperature": temperature}
        if not self._warned_sampling:
            self._warned_sampling = True
            logger.warning(
                "%s does not accept temperature; the configured value %s is not sent.",
                self.name,
                temperature,
            )
        return {}

    def _payload(
        self,
        messages: Sequence[Message],
        options: GenerationOptions | None,
        *,
        tools: ToolRequest | None = None,
    ) -> dict[str, Any]:
        options = options or GenerationOptions()
        system, rest = split_system(messages, provider=self.name)
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": option(options.max_tokens, self._max_tokens),
            **self._sampling(options),
        }
        if tools is None:
            if system is not None:
                payload["system"] = system
            payload["messages"] = [self._message(m) for m in rest]
            return payload
        payload["messages"] = self._tool_messages(rest)
        payload["tools"] = [
            {"name": spec.name, "description": spec.description, "input_schema": spec.parameters}
            for spec in tools.tools
        ]
        if self._traits.forced_tool_choice or tools.choice in ("auto", "none"):
            payload["tool_choice"] = (
                {"type": "tool", "name": tools.forced}
                if tools.forced
                else {"type": {"auto": "auto", "required": "any", "none": "none"}[tools.choice]}
            )
        else:
            # Forced tool use is a 400 on this model: steer with an instruction instead.
            payload["tool_choice"] = {"type": "auto"}
            instruction = (
                f"You must call the `{tools.forced}` tool in this reply."
                if tools.forced
                else "You must call one of the provided tools in this reply."
            )
            system = f"{system}\n\n{instruction}" if system else instruction
        if system is not None:
            payload["system"] = system
        return payload

    def _tool_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        encoded: list[dict[str, Any]] = []
        for message in messages:
            split = split_tool_message(message, provider=self.name)
            state = _replayed_content(message)
            if split.results:
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.content,
                        **({"is_error": True} if result.is_error else {}),
                    }
                    for result in split.results
                ]
                previous = encoded[-1] if encoded else None
                if (
                    previous is not None
                    and previous["role"] == "user"
                    and isinstance(previous["content"], list)
                    and previous["content"][0].get("type") == "tool_result"
                ):
                    previous["content"].extend(blocks)
                else:
                    encoded.append({"role": "user", "content": blocks})
            elif state is not None:
                encoded.append({"role": "assistant", "content": state})
            elif split.calls:
                text = [{"type": "text", "text": split.text}] if split.text else []
                encoded.append(
                    {
                        "role": "assistant",
                        "content": text
                        + [
                            {
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                # A flagged call cannot be sent as the model wrote it:
                                # tool_use input must be an object.
                                "input": call.arguments,
                            }
                            for call in split.calls
                        ],
                    }
                )
            elif split.has_media:
                encoded.append(self._message(message))
            else:
                encoded.append({"role": message.role, "content": split.text})
        return encoded

    def _message(self, message: Message) -> dict[str, Any]:
        state = _replayed_content(message)
        if state is not None:
            return {"role": "assistant", "content": state}
        if is_text_only(message):
            return {"role": message.role, "content": message_text(message, provider=self.name)}
        check_media(message, provider=self.name, content_types=self.capabilities.content_types)
        return {
            "role": message.role,
            "content": [_block(part, provider=self.name) for part in message.parts],
        }

    def _post(self, payload: dict[str, Any], *, stream: bool = False) -> requests.Response:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        kwargs: dict[str, Any] = {"stream": True} if stream else {}
        resp = requests.post(
            _API_URL,
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
            raise llm_error_from(exc, f"Anthropic request failed: {exc}") from exc

    def _complete(
        self,
        payload: dict[str, Any],
        *,
        tool: str | None = None,
        structured: bool = False,
        unwrap: bool = False,
    ) -> LLMResponse:
        data = self._send(lambda: self._post(payload).json())
        try:
            blocks = data["content"]
            tool_input = next(
                (
                    block["input"]
                    for block in blocks
                    if tool is not None
                    and block.get("type") == "tool_use"
                    and block.get("name") == tool
                ),
                None,
            )
            tool_calls: tuple[ToolCall, ...] = ()
            if tool_input is not None:
                value = tool_input.get(_WRAPPED_KEY) if unwrap else tool_input
                content = json.dumps(value, ensure_ascii=False)
            else:
                if tool is None and not structured:
                    tool_calls = tuple(
                        tool_call_from_raw(block["name"], block.get("input"), id=block.get("id"))
                        for block in blocks
                        if block.get("type") == "tool_use"
                    )
                content = "".join(
                    block["text"] for block in blocks if block.get("type") == "text"
                ).strip()
        except (KeyError, TypeError, AttributeError) as exc:
            raise LLMError(
                f"Unexpected Anthropic response shape: {type(exc).__name__}: {exc}"
            ) from exc

        if structured:
            if data.get("stop_reason") == "refusal":
                raise LLMRefusalError(f"{self.name} declined to produce structured output.")
            if unwrap:
                content = _unwrapped(content)

        # Best-effort token usage logging. Never changes the return value.
        usage = data.get("usage") or {}
        token_usage = _usage(usage.get("input_tokens"), usage.get("output_tokens"))
        if usage and token_usage is not None:
            self.last_usage = token_usage
            logger.info(
                "llm usage",
                extra={
                    "status": "ok",
                    "token_count": usage.get("output_tokens"),
                    "prompt_token_count": usage.get("input_tokens"),
                },
            )
        return LLMResponse(
            text=content,
            usage=token_usage,
            model=self.name,
            finish_reason=data.get("stop_reason"),
            tool_calls=tool_calls,
            provider_state=None if structured or tool is not None else _state(blocks),
        )


def _state(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The reply's content blocks as provider state, when it has thinking blocks that
    must be replayed; ``None`` otherwise (nothing to preserve)."""

    if not any(isinstance(b, dict) and b.get("type") in _THINKING_TYPES for b in blocks):
        return None
    return {_STATE_KEY: {"content": blocks}}


def _replayed_content(message: Message) -> list[dict[str, Any]] | None:
    if message.role != "assistant":
        return None
    state = provider_state(message, _STATE_KEY)
    content = state.get("content") if isinstance(state, dict) else None
    return content if isinstance(content, list) and content else None


def _finished_block(block: dict[str, Any]) -> dict[str, Any]:
    if "_json" not in block:
        return block
    finished = {k: v for k, v in block.items() if k != "_json"}
    try:
        value = json.loads("".join(block["_json"]) or "{}")
    except ValueError:
        value = {}
    finished["input"] = value if isinstance(value, dict) else {}
    return finished


def _unwrapped(content: str) -> str:
    try:
        value = json.loads(content)
    except ValueError:
        return content  # let the structured-output repair loop deal with it
    if isinstance(value, dict) and _WRAPPED_KEY in value:
        return json.dumps(value[_WRAPPED_KEY], ensure_ascii=False)
    return content


# JSON Schema keywords native structured outputs do not support. The constraint is
# dropped from what is sent, not from what the caller validates against.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "uniqueItems",
    }
)
_SCHEMA_MAPS = ("properties", "$defs", "definitions", "patternProperties")
_SCHEMA_LISTS = ("anyOf", "allOf", "oneOf", "prefixItems")
_SCHEMA_SINGLE = ("items", "not", "contains", "additionalProperties")


def strict_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` as native structured outputs accept it: every object schema gets
    ``additionalProperties: false`` (required by the API) unless it says otherwise,
    and numeric/length constraints the API does not support are removed. A new dict;
    the caller's schema is not modified."""

    return _strict(schema)


def _strict(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_KEYWORDS:
            continue
        if key == "minItems" and isinstance(value, int) and value > 1:
            continue
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: _strict(sub) for name, sub in value.items()}
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            out[key] = [_strict(sub) for sub in value]
        elif key in _SCHEMA_SINGLE:
            out[key] = _strict(value)
        else:
            out[key] = value
    is_object = out.get("type") == "object" or "properties" in out
    if is_object and "additionalProperties" not in out:
        out["additionalProperties"] = False
    return out


def _block(part: ContentPart, *, provider: str) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, (ImagePart, DocumentPart)):
        block: dict[str, Any] = {"type": part.type, "source": _source(part, provider=provider)}
        if isinstance(part, DocumentPart) and part.filename:
            block["title"] = part.filename
        return block
    raise UnsupportedContentError(f"{provider} cannot send content of type {part.type}.")


def _source(part: MediaPart, *, provider: str) -> dict[str, Any]:
    if part.url is not None:
        return {"type": "url", "url": http_url(part, provider=provider)}
    return {"type": "base64", "media_type": part.media_type, "data": part.base64}


def _usage(prompt: int | None, completion: int | None) -> TokenUsage | None:
    if prompt is None and completion is None:
        return None
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion)
