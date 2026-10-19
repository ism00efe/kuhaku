"""An auth-scoped, tool-using agent, as a third kuhaku tool.

Built only on ``kuhaku.core`` (tool calling, auth, audit, observability) and pydantic;
it imports nothing from the other tools. To give an agent retrieval or extraction, wrap
a ``RAG`` or ``Extractor`` call in an :class:`AgentTool` yourself.

    from kuhaku import AuthContext
    from kuhaku.tools.agent import Agent, AgentTool

    def refund(order_id: str, amount: float) -> str:
        "Refund part of an order."
        ...

    agent = Agent([AgentTool(refund, access_tags=["support"], requires_approval=True)])
    result = agent.run("refund 5 on order 42",
                       auth_context=AuthContext(identity="ada", roles=("support",)))
    result.status, result.pending, result.state
"""

from .agent import Agent, InputGuard, OutputFilter
from .replay import (
    ConsumedStateStore,
    InMemoryConsumedStateStore,
    SQLiteConsumedStateStore,
)
from .result import (
    LIMIT_STEPS,
    LIMIT_TIME,
    LIMIT_TOKENS,
    LIMIT_TOOL_CALLS,
    AgentResult,
    AgentStatus,
    CallOutcome,
    ToolExecution,
)
from .state import AgentState, AgentStateError, call_digest
from .tool import AgentTool, ArgumentsInvalid, ToolError, is_entitled

__all__ = [
    "LIMIT_STEPS",
    "LIMIT_TIME",
    "LIMIT_TOKENS",
    "LIMIT_TOOL_CALLS",
    "Agent",
    "AgentResult",
    "AgentState",
    "AgentStateError",
    "AgentStatus",
    "AgentTool",
    "ArgumentsInvalid",
    "CallOutcome",
    "ConsumedStateStore",
    "InMemoryConsumedStateStore",
    "InputGuard",
    "OutputFilter",
    "SQLiteConsumedStateStore",
    "ToolError",
    "ToolExecution",
    "call_digest",
    "is_entitled",
]
