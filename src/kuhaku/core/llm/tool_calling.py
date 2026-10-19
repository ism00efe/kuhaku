"""Tool calling: tool definitions, argument parsing, emulation, and running one call.

The pieces every provider shares, so each provider module is left with only its own
wire encoding:

- :class:`ToolSpec` -- a tool's name, description and JSON Schema parameters, written
  by hand or derived from a typed callable or a Pydantic model.
- :class:`ToolRequest` -- the validated ``tools`` + ``tool_choice`` of one call.
- :func:`tool_call_from_raw` -- one vendor tool call to a :class:`ToolCall`, with
  malformed arguments flagged on the call rather than raised or dropped.
- :func:`run_emulated_tools` -- tool calling for a provider without it, through
  structured output: the model replies ``{"tool_calls": [...]}`` or ``{"answer": ...}``.
- :func:`execute_tool_call` -- run one call against a
  registry of callables, every failure captured in the :class:`ExecutionResult`.

There is deliberately no loop here: deciding how many rounds to run, which tools a
caller may use and what to do with a failure is policy, and belongs to whatever builds
an agent on top of this.
"""

from __future__ import annotations

import inspect
import json
import re
import typing
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..media import MediaPart
from ..models import (
    ExecutionResult,
    Message,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
)
from .base import (
    LLMResponse,
    StructuredOutputError,
    StructuredResponse,
    UnsupportedContentError,
)

TOOL_CHOICES = frozenset({"auto", "none", "required"})

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class ToolSpec:
    """One tool the model may call.

    ``parameters`` is a JSON Schema whose root is an object -- the shape every vendor
    requires of a tool's arguments. ``name`` must already be what every vendor accepts
    (``[A-Za-z0-9_-]``, 1-64 characters): it is checked, never rewritten, because the
    model calls the tool by exactly this name.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise ValueError(
                f"Invalid tool name {self.name!r}: use 1-64 characters from "
                f"[A-Za-z0-9_-]."
            )
        if not isinstance(self.parameters, dict) or self.parameters.get("type") != "object":
            raise ValueError(
                f"Tool '{self.name}': parameters must be a JSON Schema with "
                f'"type": "object" at the root.'
            )

    @classmethod
    def from_model(
        cls, model: type[Any], *, name: str | None = None, description: str | None = None
    ) -> ToolSpec:
        """A tool whose arguments are the fields of a Pydantic model.

        ``name`` defaults to the model's class name, ``description`` to its docstring.
        """

        schema = dict(model.model_json_schema())
        schema.pop("title", None)
        doc = description if description is not None else _summary(model.__doc__)
        return cls(name=name or model.__name__, description=doc, parameters=schema)

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> ToolSpec:
        """A tool whose arguments are ``fn``'s parameters.

        The schema is built from the type hints by Pydantic (an unannotated parameter
        accepts anything; one with a default is optional). ``name`` defaults to
        ``fn.__name__``, ``description`` to the first paragraph of its docstring.
        ``*args``/``**kwargs`` cannot be described by a schema and are rejected.
        """

        from pydantic import create_model

        hints = typing.get_type_hints(fn)
        fields: dict[str, Any] = {}
        for param in inspect.signature(fn).parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(
                    f"Cannot build a tool from {fn.__name__}: *args/**kwargs have no "
                    f"JSON Schema."
                )
            if param.kind is param.POSITIONAL_ONLY:
                # Called as fn(**arguments), so a positional-only parameter is unreachable.
                raise TypeError(
                    f"Cannot build a tool from {fn.__name__}: parameter '{param.name}' "
                    f"is positional-only."
                )
            default = ... if param.default is param.empty else param.default
            fields[param.name] = (hints.get(param.name, Any), default)
        model = create_model(f"{fn.__name__}_arguments", **fields)
        schema = dict(model.model_json_schema())
        schema.pop("title", None)
        doc = description if description is not None else _summary(fn.__doc__)
        return cls(name=name or fn.__name__, description=doc, parameters=schema)


def _summary(doc: str | None) -> str:
    return inspect.cleandoc(doc).split("\n\n", 1)[0].strip() if doc else ""


@dataclass(frozen=True)
class ToolRequest:
    """The validated tools and tool choice of one call. Built by :meth:`build`."""

    tools: tuple[ToolSpec, ...]
    choice: str = "auto"

    @classmethod
    def build(cls, tools: Sequence[ToolSpec], tool_choice: str = "auto") -> ToolRequest:
        if isinstance(tools, ToolSpec):
            raise TypeError("tools must be a sequence of ToolSpec, not a single ToolSpec.")
        specs = tuple(tools)
        if not specs:
            raise ValueError("tools must not be empty; use chat() for a call without tools.")
        for spec in specs:
            if not isinstance(spec, ToolSpec):
                raise TypeError(
                    f"tools must contain ToolSpec objects; got {type(spec).__name__} "
                    f"(build one with ToolSpec.from_function or ToolSpec.from_model)."
                )
        names = [spec.name for spec in specs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool names: {', '.join(duplicates)}.")
        if tool_choice not in TOOL_CHOICES and tool_choice not in names:
            raise ValueError(
                f"tool_choice must be 'auto', 'none', 'required' or the name of one of "
                f"the tools ({', '.join(names)}); got {tool_choice!r}."
            )
        return cls(specs, tool_choice)

    @property
    def forced(self) -> str | None:
        """The tool the model must call, when ``choice`` names one."""

        return None if self.choice in TOOL_CHOICES else self.choice


def new_tool_call_id() -> str:
    """An id for a call whose API gave it none (Ollama, Vertex AI, emulation)."""

    return f"call_{uuid.uuid4().hex[:24]}"


def tool_call_from_raw(name: str, arguments: Any, *, id: str | None = None) -> ToolCall:
    """A :class:`ToolCall` from a vendor's name + arguments (a dict, or JSON text).

    Arguments that are not a JSON object yield a flagged call (``arguments_error``,
    ``raw_arguments``), never an exception: one malformed call must not lose the
    others in the same reply. An empty string means no arguments -- some
    OpenAI-compatible servers send that for a parameterless tool.
    """

    call_id = id or new_tool_call_id()
    if arguments is None:
        return ToolCall(name=name, arguments={}, id=call_id)
    if isinstance(arguments, dict):
        return ToolCall(name=name, arguments=arguments, id=call_id)
    if isinstance(arguments, str):
        if not arguments.strip():
            return ToolCall(name=name, arguments={}, id=call_id)
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return ToolCall(
                name=name,
                id=call_id,
                raw_arguments=arguments,
                arguments_error=f"arguments are not valid JSON ({exc.msg})",
            )
        if isinstance(parsed, dict):
            return ToolCall(name=name, arguments=parsed, id=call_id)
        return ToolCall(
            name=name,
            id=call_id,
            raw_arguments=arguments,
            arguments_error=f"arguments are a JSON {type(parsed).__name__}, not an object",
        )
    return ToolCall(
        name=name,
        id=call_id,
        raw_arguments=repr(arguments),
        arguments_error=f"arguments are a {type(arguments).__name__}, not an object",
    )


def encoded_arguments(call: ToolCall) -> str:
    """A call's arguments as JSON text, for sending the call back to the model. A
    flagged call is sent back as the model wrote it."""

    if call.raw_arguments is not None and not call.valid:
        return call.raw_arguments
    return json.dumps(call.arguments, ensure_ascii=False)


# --- reading tool-bearing messages ----------------------------------------------------


TOOL_ROLES = frozenset({"system", "user", "assistant", "tool"})

# Images and documents ride along only on user turns (the same rule as chat()); whether
# the provider accepts their type is checked by its own media encoder.
TOOL_MEDIA_ROLES = frozenset({"user"})


@dataclass(frozen=True)
class ToolMessage:
    """One message split into what a tool-calling encoder needs."""

    role: str
    text: str
    calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResultPart, ...] = ()
    has_media: bool = False
    """The message carries image/document parts: ``text`` alone does not represent it,
    so the encoder must send it through its media-aware message encoding."""


def split_tool_message(message: Message, *, provider: str) -> ToolMessage:
    """``message`` as text + tool calls + tool results, or
    :class:`UnsupportedContentError` for anything a tool-calling encoder cannot send:
    another part type, an unknown role, tool calls off an ``assistant`` message, a
    ``tool`` message that is not made of tool results, or media off a ``user`` message.
    Media parts on a user message are allowed and flagged (``has_media``); the provider's
    media encoder decides whether it can send their type."""

    if message.role not in TOOL_ROLES:
        raise UnsupportedContentError(
            f"{provider} cannot send a message with role '{message.role}'; "
            f"supported roles: {', '.join(sorted(TOOL_ROLES))}."
        )
    texts: list[str] = []
    calls: list[ToolCall] = []
    results: list[ToolResultPart] = []
    has_media = False
    for part in message.parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
        elif isinstance(part, MediaPart) and message.role in TOOL_MEDIA_ROLES:
            has_media = True
        elif isinstance(part, ToolCallPart):
            calls.append(part.call)
        elif isinstance(part, ToolResultPart):
            results.append(part)
        else:
            raise UnsupportedContentError(
                f"{provider} cannot send content of type {part.type} in a "
                f"'{message.role}' message; only text, tool calls, tool results and "
                f"(on user messages) images and documents are supported."
            )
    if calls and message.role != "assistant":
        raise UnsupportedContentError(
            f"{provider}: tool calls can only be sent on an assistant message, "
            f"not '{message.role}'."
        )
    if message.role == "tool" and (texts or not results):
        raise UnsupportedContentError(
            f"{provider}: a tool message must consist of tool results "
            f"(use Message.tool_result(tool_call_id, content))."
        )
    if results and message.role != "tool":
        raise UnsupportedContentError(
            f"{provider}: tool results can only be sent on a tool message, "
            f"not '{message.role}'."
        )
    return ToolMessage(
        message.role, "".join(texts), tuple(calls), tuple(results), has_media=has_media
    )


def tool_names_by_id(messages: Sequence[Message]) -> dict[str, str]:
    """``{call id: tool name}`` for every tool call in ``messages``."""

    return {call.id: call.name for m in messages for call in m.tool_calls}


# --- emulation ------------------------------------------------------------------------


def emulation_schema(request: ToolRequest) -> dict[str, Any]:
    """The JSON Schema an emulated reply follows: tool calls, or a direct answer."""

    forced = request.forced
    names = [forced] if forced else [spec.name for spec in request.tools]
    calls = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": names},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
        },
    }
    if request.choice == "auto":
        return {
            "type": "object",
            "properties": {"tool_calls": calls, "answer": {"type": "string"}},
        }
    return {
        "type": "object",
        "properties": {"tool_calls": {**calls, "minItems": 1}},
        "required": ["tool_calls"],
    }


def emulation_instruction(request: ToolRequest) -> str:
    tools = [
        {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
        for spec in request.tools
    ]
    if request.forced:
        rule = f'You must call the tool "{request.forced}" now; do not answer directly.'
    elif request.choice == "required":
        rule = "You must call at least one tool now; do not answer directly."
    else:
        rule = (
            "Call tools when you need them; when you can answer without them, answer "
            'directly with {"answer": "<your answer>"}.'
        )
    return (
        "You can call these tools:\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
        + '\n\nTo call tools, reply {"tool_calls": [{"name": "<tool>", "arguments": '
        "{...}}]}; the results will be sent back to you. "
        + rule
    )


def render_for_emulation(messages: Sequence[Message], *, provider: str) -> list[Message]:
    """``messages`` with tool calls and results turned into text, for a provider that
    can only send text: an assistant's calls become the JSON it would have replied
    with, a tool result a user turn naming the call it answers."""

    names = tool_names_by_id(messages)
    rendered: list[Message] = []
    for message in messages:
        split = split_tool_message(message, provider=provider)
        if split.results:
            lines = []
            for result in split.results:
                label = "error" if result.is_error else "result"
                name = result.name or names.get(result.tool_call_id, "")
                lines.append(
                    f"Tool {label} for call {result.tool_call_id}"
                    + (f" ({name})" if name else "")
                    + f":\n{result.content}"
                )
            rendered.append(Message.user("\n\n".join(lines)))
        elif split.calls:
            body = {
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in split.calls
                ]
            }
            text = (split.text + "\n" if split.text else "") + json.dumps(
                body, ensure_ascii=False
            )
            rendered.append(Message.assistant(text))
        else:
            rendered.append(message)
    return rendered


def _with_instruction(messages: list[Message], instruction: str) -> list[Message]:
    split = 0
    while split < len(messages) and messages[split].role == "system":
        split += 1
    return [*messages[:split], Message.system(instruction), *messages[split:]]


def run_emulated_tools(
    *,
    chat_turn: Callable[[tuple[Message, ...]], LLMResponse],
    structured_turn: Callable[[tuple[Message, ...], dict[str, Any]], StructuredResponse],
    messages: Sequence[Message],
    request: ToolRequest,
    provider: str,
) -> LLMResponse:
    """Tool calling for a provider without it, through structured output.

    ``tool_choice="none"`` is a plain ``chat_turn`` over the rendered conversation.
    Otherwise the tools and the reply format go into a system instruction and
    ``structured_turn`` (the provider's ``generate_structured``, native or prompted)
    asks for ``{"tool_calls": [...]}`` or ``{"answer": ...}``. With ``"auto"``, a
    reply that never parses as JSON is taken as a prose answer -- the model chose not
    to use the format -- but its token usage is lost with the parse error; with
    ``"required"`` or a named tool, :class:`StructuredOutputError` propagates.
    """

    rendered = render_for_emulation(messages, provider=provider)
    metadata = {"tool_calling": "emulated"}
    if request.choice == "none":
        response = chat_turn(tuple(rendered))
        return LLMResponse(
            text=response.text,
            usage=response.usage,
            model=response.model,
            finish_reason=response.finish_reason,
            metadata=metadata,
        )

    conversation = tuple(_with_instruction(rendered, emulation_instruction(request)))
    try:
        result = structured_turn(conversation, emulation_schema(request))
    except StructuredOutputError as exc:
        if request.choice != "auto":
            raise
        return LLMResponse(text=exc.raw.strip(), model=provider, metadata=metadata)

    data = result.data
    calls: list[ToolCall] = []
    text = result.text
    if isinstance(data, dict):
        raw_calls = data.get("tool_calls")
        if isinstance(raw_calls, list):
            for item in raw_calls:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    calls.append(tool_call_from_raw(item["name"], item.get("arguments")))
        answer = data.get("answer")
        if isinstance(answer, str):
            text = answer
        elif calls:
            text = ""
    return LLMResponse(
        text=text.strip(),
        usage=result.usage,
        model=result.model,
        finish_reason="tool_calls" if calls else result.finish_reason,
        tool_calls=tuple(calls),
        metadata=metadata,
    )


# --- running one call -----------------------------------------------------------------


def _precheck(
    call: ToolCall, registry: Mapping[str, Callable[..., Any]]
) -> tuple[Callable[..., Any] | None, ExecutionResult | None]:
    metadata = {"tool_call_id": call.id, "tool": call.name}
    if not call.valid:
        return None, ExecutionResult(
            output=None,
            success=False,
            error=f"Malformed arguments for tool '{call.name}': {call.arguments_error}",
            metadata=metadata,
        )
    fn = registry.get(call.name)
    if fn is None:
        return None, ExecutionResult(
            output=None,
            success=False,
            error=f"Unknown tool '{call.name}'.",
            metadata=metadata,
        )
    return fn, None


def _failed(call: ToolCall, exc: Exception) -> ExecutionResult:
    return ExecutionResult(
        output=None,
        success=False,
        error=f"{type(exc).__name__}: {exc}",
        metadata={"tool_call_id": call.id, "tool": call.name},
    )


def execute_tool_call(
    call: ToolCall, registry: Mapping[str, Callable[..., Any]]
) -> ExecutionResult:
    """Run ``registry[call.name](**call.arguments)``, returning its output or the
    failure -- a flagged call, an unknown tool, or an exception the callable raised --
    as an :class:`ExecutionResult`. Never raises for any of those.

    Arguments are passed as the model sent them; validating them (beyond the
    ``TypeError`` a wrong keyword raises) is the callable's business. An ``async``
    callable is reported as a failure.
    """

    fn, failure = _precheck(call, registry)
    if failure is not None:
        return failure
    assert fn is not None
    try:
        output = fn(**call.arguments)
    except Exception as exc:  # noqa: BLE001 - a tool's failure is a result, not a crash
        return _failed(call, exc)
    if inspect.isawaitable(output):
        close = getattr(output, "close", None)
        if close is not None:
            close()
        return _failed(call, TypeError(f"tool '{call.name}' is async; tools must be synchronous"))
    return ExecutionResult(output=output, metadata={"tool_call_id": call.id, "tool": call.name})


def tool_result_message(call: ToolCall, result: ExecutionResult) -> Message:
    """The ``tool`` message that sends ``result`` back to the model as the answer to
    ``call``. A non-string output is sent as JSON (``str()`` for what JSON cannot
    represent); a failure sends ``result.error`` with ``is_error`` set."""

    if not result.success:
        return Message.tool_result(
            call.id, result.error or "Tool call failed.", name=call.name, is_error=True
        )
    output = result.output
    if isinstance(output, str):
        content = output
    else:
        content = json.dumps(output, ensure_ascii=False, default=str)
    return Message.tool_result(call.id, content, name=call.name)


__all__ = [
    "TOOL_CHOICES",
    "ToolMessage",
    "ToolRequest",
    "ToolSpec",
    "emulation_instruction",
    "emulation_schema",
    "encoded_arguments",
    "execute_tool_call",
    "new_tool_call_id",
    "render_for_emulation",
    "run_emulated_tools",
    "split_tool_message",
    "tool_call_from_raw",
    "tool_names_by_id",
    "tool_result_message",
]
