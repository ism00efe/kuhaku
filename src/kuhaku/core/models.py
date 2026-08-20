"""Generic domain models shared across any tool built on kuhaku.

Plain dataclasses with no external dependencies -- the vocabulary of kuhaku's
runtime core (a tool invocation and its result, a conversational message), independent
of what any particular tool (RAG, or a future one) actually does. Tool-specific models
(e.g. the RAG tool's ``Document``/``Chunk``/``Answer``) live under their own tool
namespace -- see ``kuhaku.tools.rag.models`` -- not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    """One turn in a conversation with an LLM or tool.

    ``role`` is conventionally ``"system"``/``"user"``/``"assistant"``/``"tool"``, but is
    left as a plain string rather than an enum so a new tool can introduce its own role
    vocabulary without changing kuhaku itself.
    """

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A request to invoke one tool by name with a set of arguments."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


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
