"""Human-in-the-loop: pausing, resuming, and the ways a paused state can be abused."""

from __future__ import annotations

import json

import pytest

from kuhaku import AuthContext
from kuhaku.tools.agent import (
    Agent,
    AgentState,
    AgentStateError,
    AgentStatus,
    AgentTool,
    CallOutcome,
)

from .conftest import ScriptedToolLLM, answer, calls, no_audit, tool_results

ADA = AuthContext(identity="ada", is_authenticated=True, roles=("support",))
MALLORY = AuthContext(identity="mallory", is_authenticated=True, roles=("support",))


class Refunds:
    def __init__(self) -> None:
        self.done: list[tuple[str, float]] = []

    def refund(self, order_id: str, amount: float) -> str:
        """Refund part of an order."""

        self.done.append((order_id, amount))
        return f"refunded {amount} on {order_id}"

    def tool(self, **kwargs) -> AgentTool:
        return AgentTool(
            self.refund, requires_approval=True, access_tags=["support"], **kwargs
        )


def lookup(order_id: str) -> str:
    """Look an order up."""

    return f"order {order_id}: 20.00"


def paused(refunds: Refunds, *, replies=(), secret: str | None = "s3cret", llm=None):
    llm = llm or ScriptedToolLLM(
        calls(
            ("lookup", {"order_id": "42"}), ("refund", {"order_id": "42", "amount": 5})
        ),
        *replies,
    )
    agent = Agent(
        [lookup, refunds.tool()], llm=llm, settings=no_audit(), state_secret=secret
    )
    return agent, llm, agent.run("refund 5 on order 42", auth_context=ADA)


def test_a_call_needing_approval_pauses_the_whole_turn():
    refunds = Refunds()
    _, _, result = paused(refunds)
    assert result.status is AgentStatus.AWAITING_APPROVAL
    assert [c.name for c in result.pending] == ["refund"]
    assert refunds.done == [] and result.tool_calls == ()  # nothing in the turn ran yet
    assert result.state is not None and result.state.pending == result.pending
    assert result.messages[-1].role == "assistant"


def test_approve_then_complete():
    refunds = Refunds()
    agent, llm, result = paused(refunds, replies=[answer("Refunded 5.")])
    state_json = result.state.to_json()  # round-trips through storage

    resumed = agent.resume(state_json, {"c2": True}, auth_context=ADA)

    assert resumed.status is AgentStatus.COMPLETED and resumed.text == "Refunded 5."
    assert refunds.done == [("42", 5.0)]
    assert [e.outcome for e in resumed.tool_calls] == [CallOutcome.OK, CallOutcome.OK]
    assert resumed.trace_id == result.trace_id
    assert resumed.steps == 2
    # The model saw both results, in call order, before its final answer.
    assert [p.tool_call_id for p in tool_results(llm.requests[-1][0])] == ["c1", "c2"]


def test_deny_and_missing_decision_never_run_the_call():
    for approvals in ({"c2": False}, {}):
        refunds = Refunds()
        agent, llm, result = paused(refunds, replies=[answer("Not refunded.")])
        resumed = agent.resume(result.state, approvals, auth_context=ADA)
        assert resumed.ok and refunds.done == []
        lookup_exec, refund_exec = resumed.tool_calls
        assert lookup_exec.outcome is CallOutcome.OK
        assert refund_exec.outcome is CallOutcome.DENIED
        assert "not approved" in tool_results(llm.requests[-1][0])[-1].content


def test_resume_with_a_different_identity_is_refused():
    refunds = Refunds()
    agent, _, result = paused(refunds, replies=[answer("x")])
    with pytest.raises(AgentStateError, match="different identity"):
        agent.resume(result.state, {"c2": True}, auth_context=MALLORY)
    with pytest.raises(AgentStateError, match="different identity"):
        agent.resume(result.state, {"c2": True}, auth_context=None)
    assert refunds.done == []
    # The refused attempts did not burn the state for its owner.
    assert agent.resume(result.state, {"c2": True}, auth_context=ADA).ok


def test_tampered_state_is_refused():
    refunds = Refunds()
    agent, _, result = paused(refunds, replies=[answer("x")])
    data = json.loads(result.state.to_json())
    for message in data["payload"]["messages"]:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "tool_call" and part["call"]["name"] == "refund":
                    part["call"]["arguments"]["amount"] = 5000
    with pytest.raises(AgentStateError, match="verification"):
        agent.resume(json.dumps(data), {"c2": True}, auth_context=ADA)

    forged_identity = json.loads(result.state.to_json())
    forged_identity["payload"]["identity"]["identity"] = "mallory"
    with pytest.raises(AgentStateError, match="verification"):
        agent.resume(forged_identity, {"c2": True}, auth_context=MALLORY)
    assert refunds.done == []


def test_a_state_resumes_only_once():
    refunds = Refunds()
    agent, _, result = paused(refunds, replies=[answer("x")])
    agent.resume(result.state, {"c2": True}, auth_context=ADA)
    with pytest.raises(AgentStateError, match="already resumed"):
        agent.resume(result.state, {"c2": True}, auth_context=ADA)
    assert refunds.done == [("42", 5.0)]


def test_state_secret_decides_who_can_resume():
    refunds = Refunds()
    _, _, result = paused(refunds, secret=None)  # random per-instance key
    other = Agent(
        [lookup, refunds.tool()], llm=ScriptedToolLLM(answer("x")), settings=no_audit()
    )
    with pytest.raises(AgentStateError, match="verification"):
        other.resume(result.state, {"c2": True}, auth_context=ADA)

    _, _, result = paused(refunds, secret="shared")
    restarted = Agent(
        [lookup, refunds.tool()],
        llm=ScriptedToolLLM(answer("done")),
        settings=no_audit(),
        state_secret="shared",
    )
    assert restarted.resume(result.state.to_dict(), {"c2": True}, auth_context=ADA).ok
    assert refunds.done == [("42", 5.0)]


def test_entitlement_is_rechecked_on_resume():
    refunds = Refunds()
    agent, llm, result = paused(refunds, replies=[answer("x")])
    demoted = AuthContext(identity="ada", is_authenticated=True, roles=())
    resumed = agent.resume(result.state, {"c2": True}, auth_context=demoted)
    assert refunds.done == []
    assert resumed.tool_calls[1].outcome is CallOutcome.UNKNOWN_TOOL
    assert llm.requests[-1][1] == (
        "lookup",
    )  # the refund tool is gone for the model too


def test_approvals_must_name_pending_calls():
    refunds = Refunds()
    agent, _, result = paused(refunds, replies=[answer("x")])
    with pytest.raises(ValueError, match="not pending"):
        agent.resume(
            result.state, {"c1": True}, auth_context=ADA
        )  # c1 needs no approval
    with pytest.raises(TypeError, match="True or False"):
        agent.resume(result.state, {"c2": "yes"}, auth_context=ADA)  # type: ignore[dict-item]


def test_garbage_is_not_a_state():
    agent = Agent([lookup], llm=ScriptedToolLLM(), settings=no_audit())
    with pytest.raises(AgentStateError):
        agent.resume("not json", {}, auth_context=ADA)
    with pytest.raises(AgentStateError):
        agent.resume({"payload": {}}, {}, auth_context=ADA)


def test_invalid_arguments_to_an_approval_tool_do_not_ask_a_human():
    refunds = Refunds()
    llm = ScriptedToolLLM(
        calls(("refund", {"order_id": "42"})), answer("missing amount")
    )
    agent = Agent([refunds.tool()], llm=llm, settings=no_audit())
    result = agent.run("refund", auth_context=ADA)
    assert result.ok
    assert result.tool_calls[0].outcome is CallOutcome.INVALID_ARGUMENTS


def test_budgets_carry_across_a_resume():
    refunds = Refunds()
    llm = ScriptedToolLLM(
        calls(("refund", {"order_id": "1", "amount": 1})),
        calls(("lookup", {"order_id": "1"})),
    )
    agent = Agent([lookup, refunds.tool()], llm=llm, settings=no_audit(), max_steps=2)
    result = agent.run("x", auth_context=ADA)
    resumed = agent.resume(result.state, {"c1": True}, auth_context=ADA)
    assert resumed.status is AgentStatus.STEP_LIMIT and resumed.steps == 2


def test_state_from_dict_round_trip_keeps_messages():
    refunds = Refunds()
    _, _, result = paused(refunds)
    copy = AgentState.from_json(result.state.to_json())
    assert copy == result.state
    assert copy.messages == result.messages
