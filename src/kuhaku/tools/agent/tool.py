"""``AgentTool``: one Python callable the agent may call, plus who may see it.

The model only ever learns a tool exists through the tool definitions it is sent, so
access control here is a filter on that list (:func:`is_entitled`), applied before the
model is called -- the same "filter before ranking" rule the RAG tool applies to chunks,
with the same semantics, reimplemented rather than imported because tools never import
other tools.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from kuhaku.core.auth import AuthContext
from kuhaku.core.llm.tool_calling import ToolSpec

AUTH_CONTEXT_PARAM = "auth_context"

# Bounds what one invalid call can put back into the conversation.
_MAX_REPORTED_ERRORS = 10


class ToolError(Exception):
    """Raised by a tool to send its message to the model.

    Any other exception a tool raises reaches the model only as "the tool failed":
    its text may carry paths, hostnames or data the model has no business seeing.
    ``ToolError`` is the opt-in for a message written *for* the model ("no order with
    that id", "date must be in the future").
    """


class ArgumentsInvalid(ValueError):
    """The model's arguments do not fit the tool's schema. ``str()`` is model-facing."""


class AgentTool:
    """A callable the agent may call.

    The tool's JSON Schema comes from ``fn``'s signature (type hints + defaults) or,
    when ``args_model`` is given, from that Pydantic model -- ``fn`` is then called with
    one validated instance of it. ``name`` defaults to ``fn.__name__`` and
    ``description`` to the first paragraph of the docstring (of ``args_model`` when
    given and ``fn`` has none).

    ``access_tags`` restricts who the tool exists for, with the same rule as a tagged
    RAG document: untagged means everyone; tagged means only an ``AuthContext`` whose
    ``roles`` intersect the tags; tagged with no ``auth_context`` means nobody. A tool
    the caller is not entitled to is never sent to the model, and a call to it is
    answered exactly like a call to a tool that does not exist.

    ``requires_approval`` pauses the run before the tool executes (see
    :meth:`Agent.resume <kuhaku.tools.agent.Agent.resume>`). ``timeout`` (seconds)
    bounds how long the agent waits for the tool; a synchronous call cannot be
    interrupted, so a timed-out call keeps running in a background thread and its
    result is discarded. ``pass_auth_context`` hands the caller's ``AuthContext`` (or
    ``None``) to ``fn`` as its ``auth_context`` keyword argument -- that parameter is
    left out of the schema, so the model can neither see nor set it.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        args_model: type[BaseModel] | None = None,
        access_tags: Sequence[str] = (),
        requires_approval: bool = False,
        timeout: float | None = None,
        pass_auth_context: bool = False,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"AgentTool needs a callable; got {type(fn).__name__}.")
        tool_name = name or _fn_name(fn)
        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__call__", None)  # noqa: B004 - a callable object's own method
        ):
            raise TypeError(
                f"AgentTool '{tool_name}': async callables are not supported; the agent "
                f"loop is synchronous."
            )
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds, or None.")
        self.fn = fn
        self.access_tags: tuple[str, ...] = _validated_tags(access_tags)
        self.requires_approval = requires_approval
        self.timeout = timeout
        self.pass_auth_context = pass_auth_context
        self.args_model = args_model

        if args_model is not None:
            if pass_auth_context:
                _require_auth_param(fn, tool_name)
            doc = description
            if doc is None:
                doc = _summary(fn.__doc__) or _summary(args_model.__doc__)
            self._model: type[BaseModel] = args_model
        else:
            self._model = _arguments_model(
                fn, tool_name, exclude_auth=pass_auth_context
            )
            doc = description if description is not None else _summary(fn.__doc__)
        self.spec = ToolSpec.from_model(self._model, name=tool_name, description=doc)

    @property
    def name(self) -> str:
        return self.spec.name

    def __repr__(self) -> str:
        return f"AgentTool({self.name!r}, access_tags={list(self.access_tags)!r})"

    def bind(self, arguments: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """The ``(args, kwargs)`` to call ``fn`` with, validated and coerced.

        Raises :class:`ArgumentsInvalid` -- whose text lists what is wrong without
        echoing the values -- rather than letting a bad call reach ``fn``. Keys the
        schema does not offer are rejected, not ignored: ignoring them would let a model
        believe an argument it invented took effect.
        """

        unknown = sorted(set(arguments) - _accepted_keys(self._model))
        if unknown:
            raise ArgumentsInvalid(f"unexpected arguments: {', '.join(unknown)}")
        try:
            validated = self._model.model_validate(arguments)
        except ValidationError as exc:
            raise ArgumentsInvalid(_describe(exc)) from None
        if self.args_model is not None:
            return (validated,), {}
        return (), {
            field: getattr(validated, field) for field in type(validated).model_fields
        }

    def invoke(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        auth_context: AuthContext | None,
    ) -> Any:
        if self.pass_auth_context:
            kwargs = {**kwargs, AUTH_CONTEXT_PARAM: auth_context}
        return self.fn(*args, **kwargs)


def is_entitled(tool: AgentTool, auth_context: AuthContext | None) -> bool:
    """Whether ``tool`` exists for ``auth_context``: untagged tools for everyone, tagged
    ones only for a caller with an intersecting role, and never for ``None``."""

    if not tool.access_tags:
        return True
    if auth_context is None:
        return False
    return not set(tool.access_tags).isdisjoint(auth_context.roles)


# --- internals ------------------------------------------------------------------------


def _fn_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


def _summary(doc: str | None) -> str:
    return inspect.cleandoc(doc).split("\n\n", 1)[0].strip() if doc else ""


def _validated_tags(tags: Sequence[str]) -> tuple[str, ...]:
    if isinstance(tags, str):
        # A bare string would otherwise be read as one tag per character.
        raise TypeError(
            "access_tags must be a sequence of strings, not a single string."
        )
    out = tuple(tags)
    for tag in out:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"access_tags must be non-empty strings; got {tag!r}.")
    return out


def _require_auth_param(fn: Callable[..., Any], tool_name: str) -> None:
    if AUTH_CONTEXT_PARAM not in inspect.signature(fn).parameters:
        raise TypeError(
            f"AgentTool '{tool_name}': pass_auth_context=True needs a parameter named "
            f"'{AUTH_CONTEXT_PARAM}'."
        )


def _arguments_model(
    fn: Callable[..., Any], tool_name: str, *, exclude_auth: bool
) -> type[BaseModel]:
    """A Pydantic model of ``fn``'s parameters -- the construction
    :meth:`ToolSpec.from_function` uses, repeated here because the model is also what
    the arguments are validated against, and ``auth_context`` must be left out of it."""

    if exclude_auth:
        _require_auth_param(fn, tool_name)
    hints = typing.get_type_hints(fn)
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if exclude_auth and param.name == AUTH_CONTEXT_PARAM:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(
                f"Cannot build a tool from {tool_name}: *args/**kwargs have no JSON Schema."
            )
        if param.kind is param.POSITIONAL_ONLY:
            raise TypeError(
                f"Cannot build a tool from {tool_name}: parameter '{param.name}' is "
                f"positional-only."
            )
        default = ... if param.default is param.empty else param.default
        fields[param.name] = (hints.get(param.name, Any), default)
    model: type[BaseModel] = create_model(  # type: ignore[call-overload]
        f"{tool_name}_arguments",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields,
    )
    return model


def _accepted_keys(model: type[BaseModel]) -> set[str]:
    keys: set[str] = set()
    for field_name, info in model.model_fields.items():
        keys.add(field_name)
        for alias in (info.alias, info.validation_alias):
            if isinstance(alias, str):
                keys.add(alias)
    return keys


def _describe(exc: ValidationError) -> str:
    errors = exc.errors(include_input=False, include_url=False)
    lines = []
    for error in errors[:_MAX_REPORTED_ERRORS]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "arguments"
        lines.append(f"{location}: {error.get('msg', 'invalid')}")
    if len(errors) > _MAX_REPORTED_ERRORS:
        lines.append(f"... and {len(errors) - _MAX_REPORTED_ERRORS} more")
    return "; ".join(lines)
