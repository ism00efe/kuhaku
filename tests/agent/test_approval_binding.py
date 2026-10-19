"""Approvals bound to exact calls, state expiry, and cross-process replay protection."""

from __future__ import annotations

import json
import threading
import time

import pytest

from kuhaku import AuthContext
from kuhaku.core.config import Settings
from kuhaku.core.models import ToolCall
from kuhaku.tools.agent import (
    Agent,
    AgentStateError,
    AgentStatus,
    AgentTool,
    CallOutcome,
    ConsumedStateStore,
    InMemoryConsumedStateStore,
    SQLiteConsumedStateStore,
    call_digest,
)

from .conftest import ScriptedToolLLM, answer, calls, no_audit

ADA = AuthContext(identity="ada", is_authenticated=True, roles=("ops",))


class Bank:
    def __init__(self) -> None:
        self.sent: list[int] = []

    def transfer(self, amount: int) -> str:
        """Transfer money."""

        self.sent.append(amount)
        return f"sent {amount}"

    def tool(self) -> AgentTool:
        return AgentTool(self.transfer, requires_approval=True)


def _agent(bank: Bank, *replies, **kwargs) -> Agent:
    kwargs.setdefault("settings", no_audit())
    kwargs.setdefault("state_secret", "shared")
    return Agent([bank.tool()], llm=ScriptedToolLLM(*replies), **kwargs)


# --- finding 2: duplicate call ids --------------------------------------------------------


def test_duplicate_call_ids_are_rekeyed_so_one_approval_cannot_run_both():
    # The reviewer's repro: two calls in one turn share id "c1".
    bank = Bank()
    turn = calls(
        ToolCall(name="transfer", arguments={"amount": 1}, id="c1"),
        ToolCall(name="transfer", arguments={"amount": 1_000_000}, id="c1"),
    )
    agent = _agent(bank, turn, answer("done"))
    result = agent.run("pay", auth_context=ADA)
    assert result.status is AgentStatus.AWAITING_APPROVAL
    ids = [c.id for c in result.pending]
    assert len(set(ids)) == 2 and "c1" not in ids
    with pytest.raises(ValueError, match="not pending"):
        agent.resume(result.state, {"c1": True}, auth_context=ADA)

    small = next(c for c in result.pending if c.arguments == {"amount": 1})
    resumed = agent.resume(result.state, {small.id: True}, auth_context=ADA)
    assert bank.sent == [1]
    outcomes = {e.call.arguments["amount"]: e.outcome for e in resumed.tool_calls}
    assert outcomes == {1: CallOutcome.OK, 1_000_000: CallOutcome.DENIED}


def test_rekeyed_transcript_pairs_every_result_with_its_call():
    bank = Bank()
    turn = calls(
        ToolCall(name="transfer", arguments={"amount": 1}, id=""),
        ToolCall(name="transfer", arguments={"amount": 2}, id=""),
    )
    agent = _agent(bank, turn, answer("done"))
    result = agent.run("pay", auth_context=ADA)
    resumed = agent.resume(
        result.state, {c.id: True for c in result.pending}, auth_context=ADA
    )
    call_ids = [c.id for c in resumed.messages[-4].tool_calls]
    result_ids = [m.parts[0].tool_call_id for m in resumed.messages[-3:-1]]
    assert call_ids == result_ids and bank.sent == [1, 2]


def test_approval_can_be_pinned_to_the_call_digest():
    bank = Bank()
    agent = _agent(bank, calls(("transfer", {"amount": 7})), answer("ok"))
    result = agent.run("pay", auth_context=ADA)
    digest = result.state.pending_digests["c1"]
    assert digest == call_digest(result.pending[0])
    with pytest.raises(ValueError, match="digest mismatch"):
        agent.resume(
            result.state,
            {"c1": call_digest(ToolCall(name="transfer", arguments={"amount": 8}))},
            auth_context=ADA,
        )
    resumed = agent.resume(result.state, {"c1": digest}, auth_context=ADA)
    assert resumed.ok and bank.sent == [7]


def test_approver_identity_is_recorded_in_result_and_audit(tmp_path):
    bank = Bank()
    audit = tmp_path / "audit.jsonl"
    settings = Settings(_env_file=None, audit_enabled=True, audit_log_path=str(audit))
    agent = _agent(bank, calls(("transfer", {"amount": 3})), answer("ok"), settings=settings)
    result = agent.run("pay", auth_context=ADA)
    resumed = agent.resume(result.state, {"c1": True}, auth_context=ADA, approved_by="boss")
    [decision] = resumed.metadata["approvals"]
    assert decision == {
        "call_id": "c1",
        "digest": call_digest(result.pending[0]),
        "approved": True,
        "approved_by": "boss",
    }
    records = [json.loads(line) for line in audit.read_text("utf-8").splitlines()]
    assert records[-1]["metadata"]["approvals"] == [decision]
    assert records[-1]["identity"] == "ada"


def test_approver_defaults_to_the_resuming_identity():
    bank = Bank()
    agent = _agent(bank, calls(("transfer", {"amount": 3})), answer("ok"))
    result = agent.run("pay", auth_context=ADA)
    resumed = agent.resume(result.state, {"c1": False}, auth_context=ADA)
    assert resumed.metadata["approvals"][0]["approved_by"] == "ada"
    assert resumed.metadata["approvals"][0]["approved"] is False


# --- finding 3: expiry and replay ---------------------------------------------------------


def test_state_expires_after_state_max_age(monkeypatch):
    bank = Bank()
    agent = _agent(bank, calls(("transfer", {"amount": 3})), answer("ok"), state_max_age=60)
    result = agent.run("pay", auth_context=ADA)
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 61)
    with pytest.raises(AgentStateError, match="expired"):
        agent.resume(result.state, {"c1": True}, auth_context=ADA)
    assert bank.sent == []


def test_state_max_age_must_be_positive_and_finite():
    for bad in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            Agent([Bank().tool()], llm=ScriptedToolLLM(), settings=no_audit(), state_max_age=bad)


def test_a_version_1_state_is_refused():
    bank = Bank()
    agent = _agent(bank, calls(("transfer", {"amount": 3})))
    result = agent.run("pay", auth_context=ADA)
    from kuhaku.tools.agent.state import seal

    old = dict(result.state.payload, v=1)
    with pytest.raises(AgentStateError, match="version"):
        agent.resume(seal(old, b"shared"), {"c1": True}, auth_context=ADA)


def test_two_workers_sharing_a_sqlite_store_resume_once(tmp_path):
    # The reviewer's repro: the in-memory set is per process, so a second worker with
    # the same state_secret could resume the same approved state again.
    bank = Bank()
    path = tmp_path / "consumed.sqlite"
    worker_a = _agent(
        bank, calls(("transfer", {"amount": 9})), answer("a"),
        consumed_states=SQLiteConsumedStateStore(path),
    )
    worker_b = _agent(bank, answer("b"), consumed_states=SQLiteConsumedStateStore(path))
    result = worker_a.run("pay", auth_context=ADA)
    assert worker_a.resume(result.state, {"c1": True}, auth_context=ADA).ok
    with pytest.raises(AgentStateError, match="already resumed"):
        worker_b.resume(result.state, {"c1": True}, auth_context=ADA)
    assert bank.sent == [9]


def test_without_a_shared_store_workers_do_not_see_each_other():
    bank = Bank()
    worker_a = _agent(bank, calls(("transfer", {"amount": 9})), answer("a"))
    worker_b = _agent(bank, answer("b"))
    result = worker_a.run("pay", auth_context=ADA)
    worker_a.resume(result.state, {"c1": True}, auth_context=ADA)
    # Documented limitation of the in-memory default.
    assert worker_b.resume(result.state, {"c1": True}, auth_context=ADA).ok


def test_sqlite_store_is_set_if_absent_under_concurrency(tmp_path):
    store = SQLiteConsumedStateStore(tmp_path / "c.sqlite")
    wins: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        won = store.consume("nonce-1", time.time() + 60)
        with lock:
            wins.append(won)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(wins) == [False] * 7 + [True]


def test_sqlite_store_refuses_an_in_memory_database():
    with pytest.raises(ValueError):
        SQLiteConsumedStateStore(":memory:")


def test_in_memory_store_drops_expired_entries_only():
    store = InMemoryConsumedStateStore()
    assert store.consume("a", time.time() - 1)
    assert store.consume("b", time.time() + 60)
    assert not store.consume("b", time.time() + 60)
    assert store.consume("a", time.time() + 60)  # "a" expired, so it was pruned


def test_stores_satisfy_the_protocol(tmp_path):
    assert isinstance(InMemoryConsumedStateStore(), ConsumedStateStore)
    assert isinstance(SQLiteConsumedStateStore(tmp_path / "x.sqlite"), ConsumedStateStore)
