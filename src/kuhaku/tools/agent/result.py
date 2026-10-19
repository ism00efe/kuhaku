"""What one :meth:`Agent.run <kuhaku.tools.agent.Agent.run>` (or ``resume``) returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from kuhaku.core.llm.base import TokenUsage
from kuhaku.core.models import ExecutionResult, Message, ToolCall

if TYPE_CHECKING:
    from .state import AgentState


class AgentStatus(str, Enum):
    COMPLETED = "completed"
    """The model gave a final answer (``text``)."""

    AWAITING_APPROVAL = "awaiting_approval"
    """A call to a ``requires_approval`` tool is waiting for a human. ``pending`` lists
    the calls to decide on; pass ``state`` and the decisions to ``Agent.resume``."""

    STEP_LIMIT = "step_limit"
    """A budget ran out before a final answer -- ``limit`` says which one."""

    FAILED = "failed"
    """The run could not continue -- ``error`` says why (every provider unreachable,
    too many invalid tool calls, input rejected by the input guard)."""


class CallOutcome(str, Enum):
    """How one tool call the model made was handled -- ``ToolExecution.outcome``."""

    OK = "ok"
    ERROR = "error"
    """The tool ran and raised (or raised :class:`ToolError`)."""

    TIMEOUT = "timeout"
    UNKNOWN_TOOL = "unknown_tool"
    """No tool of that name exists *for this caller* -- a hidden tool and a nonexistent
    one are deliberately the same outcome."""

    INVALID_ARGUMENTS = "invalid_arguments"
    DENIED = "denied"
    """A ``requires_approval`` call the human did not approve; it never ran."""


# Limit names, as reported in ``AgentResult.limit``.
LIMIT_STEPS = "steps"
LIMIT_TOOL_CALLS = "tool_calls"
LIMIT_TOKENS = "tokens"
LIMIT_TIME = "time"


@dataclass(frozen=True)
class ToolExecution:
    """One tool call the model made and what happened to it.

    ``result.output`` is what the tool returned, unchanged; ``result.error`` is exactly
    the text the model was shown for a failure -- never an exception's own message
    unless the tool raised :class:`ToolError`. The exception type, when there was one,
    is ``result.metadata["error_type"]``.
    """

    call: ToolCall
    result: ExecutionResult
    outcome: CallOutcome
    duration: float = 0.0


@dataclass(frozen=True)
class AgentResult:
    """The outcome of a run.

    ``messages`` is the full transcript (system prompt, history, input, every model
    turn and every tool result as the model saw it). ``tool_calls`` covers the calls
    handled by *this* ``run``/``resume`` call; earlier ones are in ``messages``.
    ``usage`` and ``steps`` (model calls) are cumulative across resumes, because the
    budgets are. ``trace_id`` is shared by every log record and span of the run,
    resumes included.
    """

    status: AgentStatus
    text: str = ""
    messages: tuple[Message, ...] = ()
    tool_calls: tuple[ToolExecution, ...] = ()
    pending: tuple[ToolCall, ...] = ()
    state: AgentState | None = None
    usage: TokenUsage | None = None
    steps: int = 0
    limit: str | None = None
    error: str | None = None
    trace_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is AgentStatus.COMPLETED
