"""Regression tests for the serving-layer security review findings."""

from __future__ import annotations

import hashlib

import anyio
import pytest

from kuhaku.core.auth import AuthContext, ConfigAuthorizationPolicy, DefaultAllowPolicy
from kuhaku.core.exceptions import ConfigError
from kuhaku.serving import PermissionDenied, ToolService
from kuhaku.serving.limits import MAX_QUESTION_CHARS, BodySizeLimit
from kuhaku.serving.schemas import AskRequest, ExtractRequest, IngestRequest

HR = AuthContext(identity="hana", is_authenticated=True, roles=("people_ops",))


# --- ingestion: who may write what ------------------------------------------------------


def test_enable_ingest_without_a_policy_warns(served, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="kuhaku.serving.service"):
        service = ToolService(rag=served.rag, enable_ingest=True)
    assert any("without a policy" in r.getMessage() for r in caplog.records)
    # The costless rule still holds with no policy: tags must be the caller's roles.
    with pytest.raises(PermissionDenied):
        service.ingest(HR, "planted", "p.md", ["finance"])
    assert service.ingest(HR, "# Pub\n\nopen.", "pub.md").chunks_added >= 1


def test_ingest_tags_must_be_a_subset_of_the_callers_roles(served):
    service = ToolService(rag=served.rag, enable_ingest=True, policy=DefaultAllowPolicy())
    with pytest.raises(PermissionDenied):
        service.ingest(HR, "planted", "p.md", ["finance"])
    with pytest.raises(PermissionDenied):
        service.ingest(HR, "planted", "p.md", ["people_ops", "finance"])
    assert service.ingest(HR, "# Ok\n\nfine.", "ok.md", ["people_ops"]).chunks_added >= 1


def test_untagged_ingest_needs_its_own_policy_rule(served):
    policy = ConfigAuthorizationPolicy()
    policy.set_rule("rag", "ingest", ["people_ops"])
    service = ToolService(rag=served.rag, enable_ingest=True, policy=policy)
    with pytest.raises(PermissionDenied):
        service.ingest(HR, "visible to all", "pub.md")
    with pytest.raises(PermissionDenied):
        service.ingest(HR, "visible to all", "pub.md", [])
    policy.set_rule("rag", "ingest_untagged", ["people_ops"])
    assert service.ingest(HR, "# Pub\n\nvisible to all.", "pub.md").chunks_added >= 1


def test_ingestion_is_audited_without_the_text_or_filename(served):
    service = ToolService(rag=served.rag, enable_ingest=True, policy=DefaultAllowPolicy())
    result = service.ingest(HR, "# Secret body\n\nzebra-unicorn.", "zebra-file.md", ["people_ops"])
    records = [r for r in served.audit_records() if r["event_type"] == "rag_ingest"]
    assert len(records) == 1
    rec = records[0]
    assert rec["identity"] == "hana" and rec["roles"] == ["people_ops"]
    assert rec["trace_id"] == result.trace_id
    meta = rec["metadata"]
    assert meta["filename_sha256"] == hashlib.sha256(b"zebra-file.md").hexdigest()
    assert meta["chunk_count"] == result.chunks_added
    assert meta["access_tags"] == ["people_ops"]
    assert meta["size_bytes"] == len(b"# Secret body\n\nzebra-unicorn.")
    line = served.audit_path.read_text("utf-8")
    assert "zebra" not in line


# --- request size limits ----------------------------------------------------------------


def test_wire_schemas_cap_text_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AskRequest(question="x" * (MAX_QUESTION_CHARS + 1))
    with pytest.raises(ValidationError):
        AskRequest(question="q", context_text="x" * 200_000)
    with pytest.raises(ValidationError):
        ExtractRequest(text="x" * 200_000)
    with pytest.raises(ValidationError):
        IngestRequest(text="x" * 3_000_000)


async def _drive(app, messages, headers=()):
    sent: list[dict] = []
    queue = list(messages)

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": list(headers)}
    await app(scope, receive, send)
    return sent


def _inner_app(seen):
    async def app(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        seen.append(body)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


def test_body_limit_refuses_a_declared_oversized_body():
    seen: list[bytes] = []
    app = BodySizeLimit(_inner_app(seen), max_bytes=10)
    sent = anyio.run(_drive, app, [], [(b"content-length", b"11")])
    assert sent[0]["status"] == 413 and seen == []


def test_body_limit_refuses_an_undeclared_chunked_body():
    seen: list[bytes] = []
    app = BodySizeLimit(_inner_app(seen), max_bytes=10)
    chunks = [{"type": "http.request", "body": b"123456", "more_body": True}] * 3
    sent = anyio.run(_drive, app, chunks)
    assert sent[0]["status"] == 413 and seen == []


def test_body_limit_replays_an_acceptable_body():
    seen: list[bytes] = []
    app = BodySizeLimit(_inner_app(seen), max_bytes=10)
    chunks = [
        {"type": "http.request", "body": b"12345", "more_body": True},
        {"type": "http.request", "body": b"678", "more_body": False},
    ]
    sent = anyio.run(_drive, app, chunks)
    assert sent[0]["status"] == 200 and seen == [b"12345678"]


def test_http_app_enforces_its_body_limit(served):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kuhaku.serving import create_app

    client = TestClient(create_app(rag=served.rag, max_body_bytes=2_000))
    response = client.post("/rag/ask", json={"question": "x" * 5_000})
    assert response.status_code == 413
    assert response.headers["X-Trace-Id"]
    assert response.json()["error"]["code"] == "payload_too_large"
    long_question = client.post("/rag/ask", json={"question": "x" * (MAX_QUESTION_CHARS + 1)})
    assert long_question.status_code in (413, 422)


# --- MCP over HTTP ----------------------------------------------------------------------


def test_mcp_http_refuses_unauthenticated_context_that_carries_roles(served):
    # The reviewer's repro: is_authenticated defaults to False, so the old guard let
    # this through and every remote client got people_ops.
    pytest.importorskip("mcp")
    from kuhaku.serving import KuhakuMCPServer

    for context in (
        AuthContext(identity="ops-bot", roles=("people_ops",)),
        AuthContext(identity="ops-bot"),
        AuthContext(identity="", permissions=("x",)),
    ):
        with pytest.raises(ConfigError):
            KuhakuMCPServer(rag=served.rag, auth_context=context).streamable_http_app()
    KuhakuMCPServer(rag=served.rag, auth_context=AuthContext.anonymous()).streamable_http_app()


_MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def _mcp_client(served, base_url, **kwargs):
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    from kuhaku.serving import KuhakuMCPServer

    app = KuhakuMCPServer(rag=served.rag).streamable_http_app(json_response=True, **kwargs)
    return TestClient(app, base_url=base_url)


def test_mcp_http_rejects_a_foreign_host_header(served):
    with _mcp_client(served, "http://evil.example") as client:
        assert client.post("/mcp", json=_LIST, headers=_MCP_HEADERS).status_code == 421
    with _mcp_client(served, "http://127.0.0.1:8000") as client:
        assert client.post("/mcp", json=_LIST, headers=_MCP_HEADERS).status_code == 200


def test_mcp_http_rejects_a_foreign_origin(served):
    with _mcp_client(served, "http://localhost:8000") as client:
        headers = {**_MCP_HEADERS, "Origin": "http://evil.example"}
        assert client.post("/mcp", json=_LIST, headers=headers).status_code == 403


def test_mcp_http_allowed_hosts_are_configurable(served):
    with _mcp_client(served, "https://kuhaku.internal", allowed_hosts=["kuhaku.internal"]) as client:
        assert client.post("/mcp", json=_LIST, headers=_MCP_HEADERS).status_code == 200


def test_mcp_http_enforces_its_body_limit(served):
    with _mcp_client(served, "http://localhost", max_body_bytes=500) as client:
        big = {**_LIST, "params": {"pad": "x" * 2_000}}
        assert client.post("/mcp", json=big, headers=_MCP_HEADERS).status_code == 413
