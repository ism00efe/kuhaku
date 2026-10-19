"""Generic domain models shared across any tool built on kuhaku.

Plain dataclasses with no external dependencies -- the vocabulary of kuhaku's
runtime core (a tool invocation and its result, a conversational message), independent
of what any particular tool (RAG, or a future one) actually does. Tool-specific models
(e.g. the RAG tool's ``Document``/``Chunk``/``Answer``) live under their own tool
namespace -- see ``kuhaku.tools.rag.models`` -- not here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any, ClassVar


@dataclass(frozen=True)
class ContentPart:
    """One piece of a :class:`Message`'s content.

    The class is the extension seam for content that is not plain text -- an image, a
    PDF (:class:`~kuhaku.core.media.ImagePart`, :class:`~kuhaku.core.media.DocumentPart`),
    a tool call the model made, the result of running one: each is a new subclass with
    its own ``type``, added beside ``TextPart`` without touching ``Message``. A provider
    that meets a part type it cannot send raises
    :class:`~kuhaku.core.llm.base.UnsupportedContentError` rather than dropping it, so
    adding a part type never silently changes what a model sees.
    """

    type: ClassVar[str] = ""

    @cached_property
    def fingerprint(self) -> str:
        """A stable sha256 hex digest of the part's type and compared fields.

        For keys (a response cache, say) that must not embed the content itself: bytes
        are hashed as they are, never base64-encoded first, and the digest is computed
        once per part. Two parts that compare equal have the same fingerprint.
        """

        digest = hashlib.sha256(self.type.encode())
        for item in fields(self):  # type: ignore[arg-type]  # subclasses are dataclasses
            if not item.compare:
                continue
            value = getattr(self, item.name)
            digest.update(b"\x00" + item.name.encode() + b"\x00")
            if isinstance(value, bytes):
                digest.update(value)
            else:
                digest.update(json.dumps(value, sort_keys=True, default=repr).encode())
        return digest.hexdigest()


@dataclass(frozen=True)
class TextPart(ContentPart):
    text: str

    type: ClassVar[str] = "text"


@dataclass(frozen=True)
class ToolCallPart(ContentPart):
    """A tool call the model made, carried on an ``assistant`` message so the
    conversation can be sent back to the model together with the call's result."""

    call: ToolCall

    type: ClassVar[str] = "tool_call"


@dataclass(frozen=True)
class ToolResultPart(ContentPart):
    """The result of running one tool call, carried on a ``tool`` message.

    ``tool_call_id`` links it to the :class:`ToolCall` it answers. ``name`` is that
    call's tool name; it may be left empty when the call is earlier in the same
    conversation (an API that needs it -- Vertex AI -- looks it up there).
    ``is_error`` marks ``content`` as an error description rather than an output.
    """

    tool_call_id: str
    content: str
    name: str = ""
    is_error: bool = False

    type: ClassVar[str] = "tool_result"


@dataclass(frozen=True)
class Message:
    """One turn in a conversation with an LLM or tool.

    ``role`` is conventionally ``"system"``/``"user"``/``"assistant"``/``"tool"``, but is
    left as a plain string rather than an enum so a new tool can introduce its own role
    vocabulary without changing kuhaku itself. The built-in LLM providers send
    ``system``/``user``/``assistant``, plus ``tool`` (a message of
    :class:`ToolResultPart`) on the tool-calling path, and reject any other role.

    ``content`` is either a plain string (the common case, and the only form before
    v2) or a sequence of :class:`ContentPart` -- stored as a tuple. :attr:`parts` and
    :attr:`text` read both forms uniformly, so code that consumes messages should use
    those rather than ``content`` directly.
    """

    role: str
    content: str | Sequence[ContentPart]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.content, str):
            return
        parts = tuple(self.content)
        for part in parts:
            if not isinstance(part, ContentPart):
                raise TypeError(
                    f"Message content must be a str or a sequence of ContentPart; "
                    f"got an item of type {type(part).__name__}."
                )
        object.__setattr__(self, "content", parts)

    @classmethod
    def system(cls, content: str | Sequence[ContentPart]) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str | Sequence[ContentPart]) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str | Sequence[ContentPart]) -> Message:
        return cls("assistant", content)

    @classmethod
    def tool_result(
        cls, tool_call_id: str, content: str, *, name: str = "", is_error: bool = False
    ) -> Message:
        """A ``tool`` message answering the call whose id is ``tool_call_id``."""

        return cls("tool", (ToolResultPart(tool_call_id, content, name=name, is_error=is_error),))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """The calls carried by this message's tool-call parts, in order."""

        return tuple(part.call for part in self.parts if isinstance(part, ToolCallPart))

    @property
    def parts(self) -> tuple[ContentPart, ...]:
        if isinstance(self.content, str):
            return (TextPart(self.content),)
        return self.content  # type: ignore[return-value]  # normalized in __post_init__

    @property
    def text(self) -> str:
        """The text parts concatenated, in order; other part types are skipped."""

        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))


@dataclass(frozen=True)
class ToolCall:
    """A request to invoke one tool by name with a set of arguments.

    ``id`` is the identifier the model gave the call (or one generated for it when the
    API gives none); a tool result names it in ``tool_call_id``. When the model's
    arguments were not a valid JSON object the call is still surfaced, never dropped:
    ``arguments`` is empty, ``raw_arguments`` holds what the model sent and
    ``arguments_error`` says what was wrong -- check :attr:`valid` before running it.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    raw_arguments: str | None = None
    arguments_error: str | None = None

    @property
    def valid(self) -> bool:
        return self.arguments_error is None


@dataclass(frozen=True)
class ExecutionResult:
    """The outcome of running a :class:`ToolCall` (or any other unit of work).

    ``success``/``error`` let a caller distinguish "ran and produced this output" from
    "failed" without raising -- useful for orchestration code that wants to collect
    results from several tool calls before deciding how to proceed.
    """

    output: Any
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
