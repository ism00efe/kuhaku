# Serving over MCP and HTTP

`kuhaku.serving` exposes a configured `RAG` and any number of `Extractor`s to other
programs: as tools of an **MCP server** (for agents and MCP clients) and as an **HTTP
API**. Both are optional extras, imported only when used:

```bash
pip install "kuhaku[mcp]"    # the official MCP Python SDK
pip install "kuhaku[http]"   # FastAPI + uvicorn
```

You configure everything in Python — the RAG, the extractors, who may do what — and the
transports stay thin. Both apply the same rules, because both call the same
`ToolService`.

## What is exposed

| MCP tool | HTTP route | Does |
|---|---|---|
| — | `GET /healthz` | liveness; no credential needed |
| `rag_ask` | `POST /rag/ask` | `{question, context_text?}` → `{answer, citations, abstained, redactions, trace_id}` |
| `rag_ingest` | `POST /rag/ingest` | `{text, filename?, access_tags?}` → `{chunks_added, redactions, trace_id}` — **off unless `enable_ingest=True`** |
| `extract_<name>` | `POST /extract/<name>` | `{text, reference?, prior?}` → `{status, value, data, missing, issues, fields, attempts, trace_id}` |

The extraction output is typed per extractor: the OpenAPI document and the MCP tool's
`outputSchema` both carry the extraction schema's own model under `value`. A
`needs_input`, `invalid` or `failed` status is a normal result (HTTP 200), not an error —
the call worked, and the status says what the text did or did not contain.

Extractor names become URL segments and tool names, so they must match
`[A-Za-z0-9_-]{1,64}`. Pass a dict to name them, or a list to use each schema's name.

Citations carry `tag`, `document_id`, `title`, `doc_type` and `score`. The server-side
`source_path` is deliberately left out.

## Identity

kuhaku still authenticates no one. A `RequestAuthenticator` reads one header and hands
the credential to an `AuthProvider` from `kuhaku.core.auth` — `APIKeyAuthProvider`,
`JWTAuthProvider`, or your own. The `AuthContext` it returns is passed to
`rag.ask(..., auth_context=...)`, so [access control](access-control.md) applies to every
remote call, and the audit record for the request names that identity.

**Identity never comes from the request body.** Request schemas reject unknown fields,
so a body that carries `roles` or `auth_context` is a 422, not something quietly
ignored.

| Request | `required=True` (default) | `required=False` |
|---|---|---|
| valid credential | that identity | that identity |
| no credential | **401** | anonymous: sees untagged documents only |
| invalid, expired or malformed credential | **401** | **401** |

An invalid credential is rejected even when authentication is optional: downgrading a
wrong token to anonymous would hide the mistake from the client.

The default reads `Authorization: Bearer <credential>`, which fits JWTs and API keys
alike. For a bare key header use `RequestAuthenticator(provider, header="X-API-Key",
scheme=None)`.

### Authorization

Pass an `AuthorizationPolicy` (for example `ConfigAuthorizationPolicy`) as `policy=` and
it is asked before every call:

| Call | resource | action |
|---|---|---|
| ask | `rag` | `ask` |
| ingest | `rag` | `ingest` |
| extract with `<name>` | `extractor:<name>` | `extract` |

A denial is **403** for an authenticated caller and **401** for an anonymous one. A
policy that raises denies. Without a policy, every caller the authenticator lets through
may call every exposed tool — with one exception no configuration lifts: **ingestion
always requires an authenticated identity**, so an anonymous caller can never write to
the knowledge base.

## HTTP API

<!-- no-exec -->
```python
from kuhaku import RAG, Extractor
from kuhaku.core.auth import APIKeyAuthProvider, ConfigAuthorizationPolicy
from kuhaku.serving import RequestAuthenticator, create_app
from pydantic import BaseModel


class Order(BaseModel):
    """A customer placing an order."""

    product: str
    quantity: float


keys = APIKeyAuthProvider({
    "key-hr":  {"identity": "hana", "roles": ["people_ops"]},
    "key-eng": {"identity": "efe",  "roles": ["engineering"]},
})

policy = ConfigAuthorizationPolicy()
policy.set_rule("rag", "ask", ["*"])
policy.set_rule("rag", "ingest", ["people_ops"])
policy.set_rule("extractor:order", "extract", ["*"])

app = create_app(
    rag=RAG(vector_store="./kuhaku-data"),
    extractors={"order": Extractor(Order)},
    auth=RequestAuthenticator(keys),
    policy=policy,
    enable_ingest=True,
)
```

Run it with any ASGI server, for example `uvicorn my_service:app`. Then:

```bash
curl -s localhost:8000/rag/ask \
  -H 'Authorization: Bearer key-hr' -H 'Content-Type: application/json' \
  -d '{"question": "what are the salary bands"}'
```

`create_app(auth=None)` serves without authentication: every request is anonymous,
tagged documents are invisible and ingestion is impossible. A warning is logged, because
that is rarely what a deployment wants.

The handlers are synchronous and FastAPI runs them in its thread pool. Ingestion from
one server process is serialized.

## MCP server

### stdio (a local client, one user)

Over stdio there is one local user and no request headers, so the identity is
configured **statically** with `auth_context=`. Left out, it is anonymous.

<!-- no-exec -->
```python
from kuhaku import RAG
from kuhaku.core.auth import AuthContext
from kuhaku.serving import KuhakuMCPServer

server = KuhakuMCPServer(
    rag=RAG(vector_store="./kuhaku-data"),
    auth_context=AuthContext(identity="me", is_authenticated=True, roles=("people_ops",)),
)

if __name__ == "__main__":
    server.run()    # stdio
```

`is_authenticated=True` matters only for ingestion, which refuses an unauthenticated
identity. Roles decide which tagged documents the local user sees, exactly as with
`rag.ask(auth_context=...)`.

The same file can be run by the CLI, which is what an MCP client configuration usually
points at:

```bash
kuhaku-mcp --config my_server.py:server
```

`--config` takes `module:attribute` or `path/to/file.py:attribute`; the attribute is a
`KuhakuMCPServer` or a zero-argument function returning one. While serving over stdio,
anything printed to stdout (for example by `rag.load_documents`) is sent to stderr
instead, because stdout carries the protocol.

A client configuration for it looks like:

```json
{
  "mcpServers": {
    "kuhaku": {"command": "kuhaku-mcp", "args": ["--config", "/path/to/my_server.py:server"]}
  }
}
```

### Streamable HTTP (remote clients)

Over HTTP, pass `authenticator=` instead, and every tool call is authenticated from its
own HTTP request. A static authenticated `auth_context=` is refused here — it would give
every remote client the same identity — and giving both is a `ConfigError`.

<!-- no-exec -->
```python
from kuhaku import RAG
from kuhaku.core.auth import JWTAuthProvider
from kuhaku.serving import KuhakuMCPServer, RequestAuthenticator

server = KuhakuMCPServer(
    rag=RAG(vector_store="./kuhaku-data"),
    authenticator=RequestAuthenticator(JWTAuthProvider("change-me")),
)

if __name__ == "__main__":
    server.run("streamable-http", host="127.0.0.1", port=8000)   # served at /mcp
```

A request the authenticator rejects gets a plain HTTP 401 before it reaches the MCP
session. `server.streamable_http_app()` returns the Starlette app itself, to mount or
run under your own ASGI server; `kuhaku-mcp --transport streamable-http` does the same
from the command line.

## Errors

| Failure | HTTP | MCP |
|---|---|---|
| missing or invalid credential | 401 + `WWW-Authenticate` | error result `unauthenticated: …` |
| policy denial | 403 | `forbidden: …` |
| unknown extractor, ingestion disabled | 404 | `not_found: …` / `invalid_request: …` |
| ingestion conflict with another writer | 409 | `conflict: …` |
| invalid body or arguments | 422 | `invalid_request: …` |
| LLM unavailable, store unavailable | 503 | `unavailable: …` |
| anything else | 500 | `internal_error: …` |

HTTP error bodies are `{"error": {"code", "message"}, "trace_id"}`; an MCP error result
ends with `(trace_id=…)`. Messages are written for the caller and never include the
underlying exception's text — provider errors carry hostnames and paths. The exception
is logged server-side under the same trace id. A 422 reports which fields failed and
why, but does not echo their values back.

## Observability

Every HTTP response carries an `X-Trace-Id` header, and every MCP call gets its own id
in `structuredContent.trace_id`. The id is minted by the server — never taken from the
client — and is the same one on the RAG audit record, the log lines and the spans of
that request. Each call is a `serve_rag_ask` / `serve_rag_ingest` / `serve_extract`
step (span, duration metric, structured log line), and HTTP routes also record
`api_requests_total` and `api_request_duration_seconds` by endpoint and outcome.

Request bodies are never logged. RAG's audit trail covers every ask; ingestion is not
part of it, so the serving layer logs one `document ingested` line per ingest with the
filename, size and chunk count (not the content).

## Not covered

- MCP over SSE (the transport the spec has deprecated) and WebSocket are not offered.
- Conversation `history` for extraction is not accepted over the wire; `reference` and
  `prior` are.
- There is no request-size limit beyond ingestion's own `max_upload_bytes`; put one in
  front of the server (reverse proxy) if you need it.
- Rate limiting is not part of the serving layer.
