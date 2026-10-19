"""MCP server over kuhaku's tools (the official ``mcp`` SDK; the ``kuhaku[mcp]`` extra).

    from kuhaku import RAG
    from kuhaku.core.auth import AuthContext
    from kuhaku.serving import KuhakuMCPServer

    server = KuhakuMCPServer(
        rag=RAG(vector_store="./kuhaku-data"),
        auth_context=AuthContext(identity="me", is_authenticated=True, roles=("people_ops",)),
    )
    server.run()            # stdio

Tools: ``rag_ask``, ``rag_ingest`` (only with ``enable_ingest=True``) and one
``extract_<name>`` per configured extractor, whose ``outputSchema`` carries the
extraction schema's model.

Identity: over stdio there is one local user and no request headers, so the
:class:`~kuhaku.core.auth.AuthContext` is configured statically (``auth_context=``;
anonymous when omitted). Over streamable HTTP, pass ``authenticator=`` instead and every
call is authenticated from its own HTTP request; a static context is refused there,
because it would hand one identity to every remote client.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from kuhaku.core.auth import AuthContext, AuthorizationPolicy
from kuhaku.core.config import Settings
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.observability import bind_trace_id, new_trace_id

from .auth import RequestAuthenticator
from .errors import AuthenticationRequired, InvalidRequest, ServingError
from .limits import DEFAULT_MAX_BODY_BYTES, BodySizeLimit
from .schemas import (
    AskRequest,
    AskResponse,
    ExtractRequest,
    IngestRequest,
    IngestResponse,
    extraction_response_model,
    json_schema,
)
from .service import ServableRAG, ToolService

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server

    from kuhaku.tools.extraction import Extractor

logger = logging.getLogger(__name__)

TRANSPORTS = ("stdio", "streamable-http")


def _import_mcp() -> Any:
    try:
        import mcp
    except ImportError as exc:
        raise ImportError("The kuhaku MCP server needs the MCP SDK: pip install 'kuhaku[mcp]'") from exc
    return mcp


@dataclass(frozen=True)
class _ToolSpec:
    name: str
    description: str
    request_model: type[BaseModel]
    output_schema: dict[str, Any]
    read_only: bool
    # (auth, validated request) -> (text for the model, structured payload)
    run: Callable[[AuthContext, Any], tuple[str, dict[str, Any]]]


class KuhakuMCPServer:
    """kuhaku's tools as an MCP server.

    ``rag``, ``extractors``, ``policy`` and ``enable_ingest`` are as for
    :class:`~kuhaku.serving.service.ToolService`. ``auth_context`` is the static
    identity for stdio; ``authenticator`` authenticates each streamable-HTTP request.
    Giving both is a :class:`~kuhaku.core.exceptions.ConfigError`.
    """

    def __init__(
        self,
        *,
        rag: ServableRAG | None = None,
        extractors: Mapping[str, Extractor[Any]] | Sequence[Extractor[Any]] | None = None,
        auth_context: AuthContext | None = None,
        authenticator: RequestAuthenticator | None = None,
        policy: AuthorizationPolicy | None = None,
        enable_ingest: bool = False,
        name: str = "kuhaku",
        instructions: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        if auth_context is not None and authenticator is not None:
            raise ConfigError(
                "Pass auth_context= (stdio, one local user) or authenticator= "
                "(streamable HTTP, per request), not both."
            )
        self.service = ToolService(
            rag=rag,
            extractors=extractors,
            policy=policy,
            enable_ingest=enable_ingest,
            settings=settings,
        )
        self.auth_context = auth_context or AuthContext.anonymous()
        self.authenticator = authenticator
        self.name = name
        self.instructions = instructions
        self._specs = {spec.name: spec for spec in self._tool_specs()}

    # --- tool definitions ----------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        return list(self._specs)

    def _tool_specs(self) -> list[_ToolSpec]:
        service = self.service
        specs: list[_ToolSpec] = []
        if service.rag is not None:
            specs.append(
                _ToolSpec(
                    name="rag_ask",
                    description=(
                        "Answer a question from the knowledge base, with citations. "
                        "'abstained' is true when nothing the caller may see was found."
                    ),
                    request_model=AskRequest,
                    output_schema=json_schema(AskResponse, output=True),
                    read_only=True,
                    run=lambda auth, req: _ask_result(
                        service.ask(auth, req.question, req.context_text)
                    ),
                )
            )
        if service.enable_ingest:
            specs.append(
                _ToolSpec(
                    name="rag_ingest",
                    description="Add one document to the knowledge base.",
                    request_model=IngestRequest,
                    output_schema=json_schema(IngestResponse, output=True),
                    read_only=False,
                    run=lambda auth, req: _ingest_result(
                        service.ingest(auth, req.text, req.filename, req.access_tags)
                    ),
                )
            )
        for name, extractor in service.extractors.items():
            specs.append(self._extract_spec(name, extractor))
        return specs

    def _extract_spec(self, name: str, extractor: Extractor[Any]) -> _ToolSpec:
        schema = extractor.schema
        description = f"Extract a {schema.name} from text."
        if schema.description:
            description = f"{description} {schema.description}"
        service = self.service

        def run(auth: AuthContext, req: ExtractRequest) -> tuple[str, dict[str, Any]]:
            payload = service.extract(name, auth, req.text, reference=req.reference, prior=req.prior)
            return json.dumps(payload, ensure_ascii=False), payload

        return _ToolSpec(
            name=f"extract_{name}",
            description=description,
            request_model=ExtractRequest,
            output_schema=json_schema(extraction_response_model(schema), output=True),
            read_only=True,
            run=run,
        )

    # --- the SDK server ------------------------------------------------------------

    def build_server(self) -> Server:
        """A fresh low-level ``mcp`` ``Server`` with this object's tools registered."""

        _import_mcp()
        import anyio
        from mcp import types
        from mcp.server.lowlevel import Server

        server: Server = Server(self.name, version=_version(), instructions=self.instructions)
        tools = [
            types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=json_schema(spec.request_model),
                outputSchema=spec.output_schema,
                annotations=types.ToolAnnotations(
                    readOnlyHint=spec.read_only,
                    destructiveHint=False if not spec.read_only else None,
                    openWorldHint=False,
                ),
            )
            for spec in self._specs.values()
        ]

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return tools

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
            trace_id = new_trace_id()
            try:
                spec = self._specs.get(name)
                if spec is None:
                    raise InvalidRequest(f"Unknown tool '{name}'.")
                auth = self._auth_for_current_request(server)
                try:
                    request = spec.request_model.model_validate(arguments)
                except ValidationError as exc:
                    fields = sorted({".".join(str(p) for p in e["loc"]) or "(root)" for e in exc.errors()})
                    raise InvalidRequest(f"Invalid arguments: {', '.join(fields)}.") from exc

                def work() -> tuple[str, dict[str, Any]]:
                    # Bound inside the worker thread so the id is the one every log line,
                    # span and audit record of this call carries.
                    with bind_trace_id(trace_id):
                        return spec.run(auth, request)

                text, payload = await anyio.to_thread.run_sync(work)
            except ServingError as exc:
                return _error_result(types, exc, trace_id)
            except Exception as exc:  # the SDK would echo str(exc) to the client
                logger.error("MCP tool call failed", exc_info=exc, extra={"trace_id": trace_id})
                return _error_result(types, ServingError(), trace_id)
            return [types.TextContent(type="text", text=text)], payload

        return server

    def _auth_for_current_request(self, server: Server) -> AuthContext:
        if self.authenticator is None:
            return self.auth_context
        try:
            request = server.request_context.request
        except LookupError:
            request = None
        headers = getattr(request, "headers", None)
        if headers is None:
            # An authenticator with no HTTP request to read (stdio, in-memory): there is
            # no credential, so fail closed rather than fall back to anything.
            raise AuthenticationRequired()
        return self.authenticator.authenticate(headers)

    # --- transports ----------------------------------------------------------------

    async def run_stdio_async(self) -> None:
        """Serve over stdio until the client disconnects."""

        if self.authenticator is not None:
            raise ConfigError(
                "authenticator= reads HTTP headers, which stdio does not have; configure "
                "the local user with auth_context= instead."
            )
        _import_mcp()
        import anyio
        from mcp.server.stdio import stdio_server

        # stdout is the protocol channel: anything else printed to it (RAG.load_documents
        # prints progress) would corrupt the stream, so it goes to stderr while serving.
        protocol_out = anyio.wrap_file(_utf8_text(sys.__stdout__))
        server = self.build_server()
        with contextlib.redirect_stdout(sys.stderr):
            async with stdio_server(stdout=protocol_out) as (read_stream, write_stream):
                await server.run(read_stream, write_stream, server.create_initialization_options())

    def streamable_http_app(
        self,
        *,
        path: str = "/mcp",
        stateless: bool = True,
        json_response: bool = False,
        allowed_hosts: Sequence[str] | None = None,
        allowed_origins: Sequence[str] | None = None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    ) -> Any:
        """A Starlette ASGI app serving MCP's streamable HTTP transport at ``path``.

        A request the authenticator rejects gets a plain HTTP 401 before it reaches the
        MCP session; an accepted one is authenticated again inside each tool call, from
        the same headers, so every call carries its own identity. Without an
        authenticator every request is anonymous (a warning is logged), and the server
        must have been built without an identity: any ``auth_context`` other than
        :meth:`AuthContext.anonymous` is a :class:`ConfigError`, because it would be
        handed to every remote client.

        DNS-rebinding protection is always on: the ``Host`` header must match
        ``allowed_hosts`` and a present ``Origin`` must match ``allowed_origins``
        (``"name:*"`` allows any port). Both default to loopback only (``localhost``,
        ``127.0.0.1``, ``[::1]``); a server reachable under another name must list it.
        ``max_body_bytes`` caps each request body (413 above it).
        """

        _import_mcp()
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from mcp.server.transport_security import TransportSecuritySettings
        from starlette.applications import Starlette
        from starlette.datastructures import Headers
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        if self.authenticator is None:
            if not _is_anonymous(self.auth_context):
                # Checking is_authenticated alone let AuthContext(identity=..., roles=...)
                # (is_authenticated defaults to False) through, and every remote client
                # then carried those roles.
                raise ConfigError(
                    "A static auth_context= would give every remote client the same "
                    "identity and roles; over HTTP, pass authenticator= instead."
                )
            logger.warning(
                "kuhaku MCP over HTTP without an authenticator: every request is "
                "anonymous and sees only untagged documents"
            )
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(_LOOPBACK_HOSTS if allowed_hosts is None else allowed_hosts),
            allowed_origins=list(
                _LOOPBACK_ORIGINS if allowed_origins is None else allowed_origins
            ),
        )
        manager = StreamableHTTPSessionManager(
            app=self.build_server(),
            stateless=stateless,
            json_response=json_response,
            security_settings=security,
        )
        authenticator = self.authenticator

        class _Endpoint:
            async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                if authenticator is not None:
                    try:
                        authenticator.authenticate(Headers(scope=scope))
                    except AuthenticationRequired as exc:
                        response = JSONResponse(
                            {"error": {"code": exc.code, "message": exc.message}},
                            status_code=exc.status_code,
                            headers={"WWW-Authenticate": authenticator.challenge},
                        )
                        await response(scope, receive, send)
                        return
                await manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(_app: Any) -> AsyncIterator[None]:
            async with manager.run():
                yield

        app = Starlette(
            routes=[Route(path, endpoint=_Endpoint(), methods=["GET", "POST", "DELETE"])],
            lifespan=lifespan,
        )
        app.add_middleware(BodySizeLimit, max_bytes=max_body_bytes)
        return app

    def run(
        self,
        transport: str = "stdio",
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/mcp",
        allowed_hosts: Sequence[str] | None = None,
        allowed_origins: Sequence[str] | None = None,
    ) -> None:
        """Serve until interrupted. ``transport`` is ``"stdio"`` or ``"streamable-http"``.

        For streamable HTTP, ``allowed_hosts``/``allowed_origins`` are as for
        :meth:`streamable_http_app`; when ``allowed_hosts`` is omitted and ``host`` is a
        specific address (not a wildcard bind), that address is allowed besides
        loopback.
        """

        if transport == "stdio":
            import anyio

            anyio.run(self.run_stdio_async)
        elif transport == "streamable-http":
            import uvicorn

            if allowed_hosts is None and host not in ("0.0.0.0", "::", ""):
                named = f"[{host}]" if ":" in host else host
                allowed_hosts = [*_LOOPBACK_HOSTS, named, f"{named}:*"]
            app = self.streamable_http_app(
                path=path, allowed_hosts=allowed_hosts, allowed_origins=allowed_origins
            )
            uvicorn.run(app, host=host, port=port)
        else:
            raise ConfigError(f"transport must be one of {TRANSPORTS}, got {transport!r}.")


_LOOPBACK_HOSTS = ("localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*", "[::1]", "[::1]:*")
_LOOPBACK_ORIGINS = (
    "http://localhost",
    "http://localhost:*",
    "http://127.0.0.1",
    "http://127.0.0.1:*",
    "http://[::1]",
    "http://[::1]:*",
)


def _is_anonymous(context: AuthContext) -> bool:
    return (
        not context.is_authenticated
        and not context.identity
        and not context.roles
        and not context.permissions
    )


def serve_mcp(
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    **kwargs: Any,
) -> None:
    """Build a :class:`KuhakuMCPServer` from ``kwargs`` and serve it."""

    KuhakuMCPServer(**kwargs).run(transport, host=host, port=port)


def _ask_result(response: AskResponse) -> tuple[str, dict[str, Any]]:
    text = response.answer
    if response.citations:
        sources = "\n".join(f"[{c.tag}] {c.title}" for c in response.citations)
        text = f"{text}\n\nSources:\n{sources}"
    return text, response.model_dump(mode="json")


def _ingest_result(response: IngestResponse) -> tuple[str, dict[str, Any]]:
    return f"Indexed {response.chunks_added} chunk(s).", response.model_dump(mode="json")


def _error_result(types: Any, exc: ServingError, trace_id: str) -> Any:
    return types.CallToolResult(
        content=[
            types.TextContent(type="text", text=f"{exc.code}: {exc.message} (trace_id={trace_id})")
        ],
        isError=True,
    )


def _utf8_text(stream: Any) -> Any:
    from io import TextIOWrapper

    # Re-wrapped for the same reason the SDK does: Windows consoles default to a legacy
    # code page.
    return TextIOWrapper(stream.buffer, encoding="utf-8")


def _version() -> str:
    from kuhaku import __version__

    return __version__
