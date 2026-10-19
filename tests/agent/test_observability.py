"""Audit records and spans for agent runs."""

from __future__ import annotations

import json

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from kuhaku import AuthContext
from kuhaku.core.config import Settings
from kuhaku.tools.agent import Agent, AgentTool

from .conftest import ScriptedToolLLM, answer, calls, no_audit

ADA = AuthContext(identity="ada", is_authenticated=True, roles=("support",))


def refund(order_id: str, amount: float) -> str:
    """Refund part of an order."""

    return "refund-receipt-SECRET-OUTPUT"


def lookup(order_id: str) -> str:
    """Look an order up."""

    return "lookup-SECRET-OUTPUT"


def test_audit_record_has_identity_and_tool_names_but_no_arguments_or_outputs(tmp_path):
    log = tmp_path / "audit.jsonl"
    llm = ScriptedToolLLM(
        calls(
            ("lookup", {"order_id": "ARG-VALUE-1"}),
            ("refund", {"order_id": "ARG-VALUE-2", "amount": 7}),
        ),
        answer("done"),
    )
    agent = Agent(
        [lookup, AgentTool(refund, requires_approval=True)],
        llm=llm,
        settings=Settings(audit_enabled=True, audit_log_path=str(log)),
    )
    paused = agent.run("please refund ada@example.com", auth_context=ADA)
    done = agent.resume(paused.state, {"c2": True}, auth_context=ADA)

    raw = log.read_text(encoding="utf-8")
    for secret in ("ARG-VALUE", "SECRET-OUTPUT", "ada@example.com"):
        assert secret not in raw
    first, second = [json.loads(line) for line in raw.splitlines()]
    # The input is kept the way RAG keeps a question: hashed, plus a sanitized prefix.
    assert first["input_redacted"] == "please refund [EMAIL]"
    assert first["event_type"] == second["event_type"] == "agent_run"
    assert first["identity"] == "ada" and first["roles"] == ["support"]
    assert first["action"] == "awaiting_approval"
    assert first["metadata"]["tools_pending"] == ["refund"]
    assert first["metadata"]["tools_called"] == []
    assert second["action"] == "completed"
    assert second["metadata"]["tools_called"] == ["lookup", "refund"]
    assert first["trace_id"] == second["trace_id"] == done.trace_id


def test_unknown_tool_names_are_not_echoed_into_the_audit_record(tmp_path):
    log = tmp_path / "audit.jsonl"
    llm = ScriptedToolLLM(calls(("payroll_export", {})), answer("no"))
    agent = Agent([lookup], llm=llm, settings=Settings(audit_log_path=str(log)))
    agent.run("x", auth_context=ADA)
    record = json.loads(log.read_text(encoding="utf-8"))
    assert "payroll_export" not in json.dumps(record)
    assert record["metadata"]["call_outcomes"] == {"unknown_tool": 1}


def test_spans_per_step_and_per_tool():
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    llm = ScriptedToolLLM(calls(("lookup", {"order_id": "1"})), answer("ok"))
    Agent([lookup], llm=llm, settings=no_audit()).run("x", auth_context=ADA)

    names = [span.name for span in exporter.get_finished_spans()]
    assert names.count("agent.step") == 2
    assert names.count("agent.tool.lookup") == 1
    assert "agent" in names
