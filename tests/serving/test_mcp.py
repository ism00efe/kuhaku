"""The MCP server through the SDK's in-memory client session, and its streamable-HTTP
app through Starlette's TestClient. Skipped without the ``mcp`` extra."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

import anyio
import jsonschema
from mcp.shared.memory import create_connected_server_and_client_session

from kuhaku.core.auth import AuthContext
from kuhaku.core.exceptions import ConfigError
from kuhaku.serving import KuhakuMCPServer, RequestAuthenticator
from tests.extraction.conftest import down

from .conftest import ENG_KEY, HR_KEY, order_extractor

GOOD = {"data": {"product": "tea", "quantity": 5}}
HR = AuthContext(identity="hana", is_authenticated=True, roles=("people_ops",))


def _session_call(server: KuhakuMCPServer, steps):
    """Run ``steps(session)`` against ``server`` over the in-memory transport."""

    async def main():
        async with create_connected_server_and_client_session(server.build_server()) as session:
            return await steps(session)

    return anyio.run(main)


def test_tools_are_listed_with_schemas(served):
    server = KuhakuMCPServer(rag=served.rag, extractors={"order": order_extractor()})

    async def steps(session):
        return (await session.list_tools()).tools

    tools = {t.name: t for t in _session_call(server, steps)}
    assert set(tools) == {"rag_ask", "extract_order"}  # rag_ingest is off by default
    assert tools["rag_ask"].inputSchema["required"] == ["question"]
    assert tools["rag_ask"].annotations.readOnlyHint is True
    order_out = tools["extract_order"].outputSchema
    assert set(order_out["$defs"]["Order"]["properties"]) == {"product", "quantity"}
    assert "customer placing an order" in tools["extract_order"].description


def test_rag_ask_uses_the_static_identity(served):
    server = KuhakuMCPServer(rag=served.rag, auth_context=HR)

    async def steps(session):
        return await session.call_tool("rag_ask", {"question": "salary bands"})

    result = _session_call(server, steps)
    assert not result.isError
    assert result.structuredContent["abstained"] is False
    assert "Salary bands range" in served.llm.last_user
    tid = result.structuredContent["trace_id"]
    audit = [r for r in served.audit_records() if r["trace_id"] == tid]
    assert audit and audit[-1]["identity"] == "hana"


def test_default_static_identity_is_anonymous(served):
    server = KuhakuMCPServer(rag=served.rag)

    async def steps(session):
        return await session.call_tool("rag_ask", {"question": "salary bands"})

    assert not _session_call(server, steps).isError
    assert "Salary bands range" not in served.llm.last_user


def test_roles_in_arguments_are_rejected(served):
    server = KuhakuMCPServer(rag=served.rag)

    async def steps(session):
        return await session.call_tool("rag_ask", {"question": "q", "roles": ["people_ops"]})

    result = _session_call(server, steps)
    assert result.isError
    assert served.llm.last_user is None


def test_extract_tool_returns_schema_valid_structured_output(served):
    server = KuhakuMCPServer(extractors={"order": order_extractor(GOOD)})

    async def steps(session):
        tools = {t.name: t for t in (await session.list_tools()).tools}
        result = await session.call_tool("extract_order", {"text": "5 boxes of tea"})
        return tools["extract_order"], result

    tool, result = _session_call(server, steps)
    assert not result.isError
    assert result.structuredContent["value"] == {"product": "tea", "quantity": 5.0}
    jsonschema.validate(result.structuredContent, tool.outputSchema)


def test_ingest_tool_when_enabled(served):
    server = KuhakuMCPServer(rag=served.rag, auth_context=HR, enable_ingest=True)

    async def steps(session):
        ingest = await session.call_tool(
            "rag_ingest",
            {"text": "# Holidays\n\nThe office closes in August.", "filename": "h.md",
             "access_tags": ["people_ops"]},
        )
        ask = await session.call_tool("rag_ask", {"question": "office close"})
        return ingest, ask

    ingest, ask = _session_call(server, steps)
    assert not ingest.isError and ingest.structuredContent["chunks_added"] >= 1
    assert not ask.isError
    assert "office closes in August" in served.llm.last_user


def test_anonymous_stdio_identity_cannot_ingest(served):
    server = KuhakuMCPServer(rag=served.rag, enable_ingest=True)

    async def steps(session):
        return await session.call_tool("rag_ingest", {"text": "x"})

    result = _session_call(server, steps)
    assert result.isError
    assert result.content[0].text.startswith("unauthenticated:")


def test_errors_are_mapped_without_internals():
    server = KuhakuMCPServer(extractors={"order": order_extractor(down("10.9.9.9 refused"))})

    async def steps(session):
        return await session.call_tool("extract_order", {"text": "5 tea"})

    result = _session_call(server, steps)
    assert result.isError
    text = result.content[0].text
    assert text.startswith("unavailable:")
    assert "trace_id=" in text
    assert "10.9.9.9" not in text


def test_authenticator_without_http_request_fails_closed(served, keys):
    server = KuhakuMCPServer(rag=served.rag, authenticator=RequestAuthenticator(keys))

    async def steps(session):
        return await session.call_tool("rag_ask", {"question": "q"})

    result = _session_call(server, steps)
    assert result.isError
    assert served.llm.last_user is None


def test_static_identity_and_authenticator_are_exclusive(served, keys):
    with pytest.raises(ConfigError):
        KuhakuMCPServer(rag=served.rag, auth_context=HR, authenticator=RequestAuthenticator(keys))


def test_stdio_refuses_an_authenticator(served, keys):
    server = KuhakuMCPServer(rag=served.rag, authenticator=RequestAuthenticator(keys))
    with pytest.raises(ConfigError):
        anyio.run(server.run_stdio_async)


def test_http_refuses_a_static_authenticated_identity(served):
    with pytest.raises(ConfigError):
        KuhakuMCPServer(rag=served.rag, auth_context=HR).streamable_http_app()


# --- streamable HTTP -----------------------------------------------------------------


def _rpc(name: str, arguments: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


_MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def http_client(served, keys):
    from starlette.testclient import TestClient

    server = KuhakuMCPServer(rag=served.rag, authenticator=RequestAuthenticator(keys))
    app = server.streamable_http_app(json_response=True)
    with TestClient(app, base_url="http://localhost:8000") as client:
        yield client


def test_streamable_http_rejects_missing_credential(http_client, served):
    response = http_client.post("/mcp", json=_rpc("rag_ask", {"question": "q"}), headers=_MCP_HEADERS)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert served.llm.last_user is None


def test_streamable_http_authenticates_each_call(http_client, served):
    headers = {**_MCP_HEADERS, "Authorization": f"Bearer {HR_KEY}"}
    response = http_client.post("/mcp", json=_rpc("rag_ask", {"question": "salary"}), headers=headers)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["isError"] is False
    assert "Salary bands range" in served.llm.last_user
    tid = result["structuredContent"]["trace_id"]
    audit = [r for r in served.audit_records() if r["trace_id"] == tid]
    assert audit and audit[-1]["identity"] == "hana"

    headers = {**_MCP_HEADERS, "Authorization": f"Bearer {ENG_KEY}"}
    http_client.post("/mcp", json=_rpc("rag_ask", {"question": "salary"}), headers=headers)
    assert "Salary bands range" not in served.llm.last_user
