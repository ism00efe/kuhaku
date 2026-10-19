"""HTTP API over kuhaku's tools (FastAPI; the ``kuhaku[http]`` extra).

    from kuhaku import RAG
    from kuhaku.core.auth import APIKeyAuthProvider
    from kuhaku.serving import RequestAuthenticator, create_app

    keys = APIKeyAuthProvider({"k-123": {"identity": "ada", "roles": ["people_ops"]}})
    app = create_app(rag=RAG(), auth=RequestAuthenticator(keys))
    # uvicorn module:app

Routes: ``GET /healthz``, ``POST /rag/ask``, ``POST /rag/ingest`` (only with
``enable_ingest=True``), ``POST /extract/{name}`` (one route per configured extractor).
Every response carries ``X-Trace-Id``; errors are ``{"error": {"code", "message"},
"trace_id"}`` and never include exception text from the tool.
"""

# No `from __future__ import annotations` here: FastAPI resolves route annotations at
# registration, and `Request` is a local name inside create_app.
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from kuhaku.core.auth import AuthContext, AuthorizationPolicy
from kuhaku.core.observability import (
    API_REQUEST_DURATION,
    API_REQUESTS,
    bind_trace_id,
    get_trace_id,
)

from .auth import RequestAuthenticator
from .errors import AuthenticationRequired, ServingError
from .limits import DEFAULT_MAX_BODY_BYTES, BodySizeLimit
from .schemas import (
    AskRequest,
    AskResponse,
    ExtractRequest,
    IngestRequest,
    IngestResponse,
    extraction_response_model,
)
from .service import ServableRAG, ToolService

if TYPE_CHECKING:
    from fastapi import FastAPI

    from kuhaku.core.config import Settings
    from kuhaku.tools.extraction import Extractor

logger = logging.getLogger(__name__)

TRACE_ID_HEADER = "X-Trace-Id"

T = TypeVar("T")


def _import_fastapi() -> Any:
    try:
        import fastapi
    except ImportError as exc:
        raise ImportError(
            "The kuhaku HTTP API needs FastAPI: pip install 'kuhaku[http]'"
        ) from exc
    return fastapi


def create_app(
    *,
    rag: ServableRAG | None = None,
    extractors: "Mapping[str, Extractor[Any]] | Sequence[Extractor[Any]] | None" = None,
    auth: RequestAuthenticator | None = None,
    policy: AuthorizationPolicy | None = None,
    enable_ingest: bool = False,
    title: str = "kuhaku",
    settings: "Settings | None" = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> "FastAPI":
    """Build the FastAPI app serving ``rag`` and ``extractors``.

    ``auth`` authenticates each request from its headers (see
    :class:`RequestAuthenticator`). ``None`` means no authentication at all: every
    request is anonymous, so tagged documents are invisible and ingestion is impossible
    -- a warning is logged, because that is rarely what a deployment wants.
    ``policy``/``enable_ingest``/``settings`` are as for
    :class:`~kuhaku.serving.service.ToolService` (ingestion needs a policy).
    ``max_body_bytes`` caps every request body (413 above it), before it is parsed.
    """

    fastapi = _import_fastapi()
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    service = ToolService(
        rag=rag,
        extractors=extractors,
        policy=policy,
        enable_ingest=enable_ingest,
        settings=settings,
    )
    if auth is None:
        logger.warning(
            "kuhaku HTTP API built without an authenticator: every request is anonymous"
        )

    app = fastapi.FastAPI(title=title, version=_version())
    # Added first, so it runs inside the trace-id middleware and a 413 still carries
    # X-Trace-Id.
    app.add_middleware(BodySizeLimit, max_bytes=max_body_bytes)
    app.add_middleware(_TraceIdMiddleware)

    def authenticate(request: Request) -> AuthContext:
        if auth is None:
            return AuthContext.anonymous()
        return auth.authenticate(request.headers)

    auth_dependency = fastapi.Depends(authenticate)

    @app.exception_handler(ServingError)
    async def _serving_error(request: Any, exc: ServingError) -> Any:
        headers = {}
        if isinstance(exc, AuthenticationRequired):
            headers["WWW-Authenticate"] = auth.challenge if auth is not None else "Bearer"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Any, exc: RequestValidationError) -> Any:
        # FastAPI's default 422 echoes the offending input back; only the location and
        # the reason are returned here.
        details = [
            {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
            for err in exc.errors()
        ]
        body = _error_body("invalid_request", "The request body is invalid.")
        body["error"]["details"] = details
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Any, exc: StarletteHTTPException) -> Any:
        # Unknown routes (an extractor that isn't configured, ingestion turned off) and
        # wrong methods get the same body shape as every other error.
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(code, str(exc.detail)),
            headers=getattr(exc, "headers", None),
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    if service.rag is not None:

        @app.post("/rag/ask", response_model=AskResponse)
        def rag_ask(body: AskRequest, context: AuthContext = auth_dependency) -> AskResponse:
            return _measured(
                "/rag/ask", lambda: service.ask(context, body.question, body.context_text)
            )

    if service.enable_ingest:

        @app.post("/rag/ingest", response_model=IngestResponse)
        def rag_ingest(body: IngestRequest, context: AuthContext = auth_dependency) -> IngestResponse:
            return _measured(
                "/rag/ingest",
                lambda: service.ingest(context, body.text, body.filename, body.access_tags),
            )

    for name, extractor in service.extractors.items():
        _add_extract_route(app, service, name, extractor, auth_dependency)

    return app


def _add_extract_route(
    app: Any, service: ToolService, name: str, extractor: "Extractor[Any]", auth_dependency: Any
) -> None:
    path = f"/extract/{name}"

    def extract(body: ExtractRequest, context: AuthContext = auth_dependency) -> dict[str, Any]:
        return _measured(
            "/extract/{name}",
            lambda: service.extract(
                name, context, body.text, reference=body.reference, prior=body.prior
            ),
        )

    app.add_api_route(
        path,
        extract,
        methods=["POST"],
        response_model=extraction_response_model(extractor.schema),
        name=f"extract_{name}",
        summary=f"Extract {extractor.schema.name}",
        description=extractor.schema.description or None,
    )


def _measured(endpoint: str, call: Callable[[], T]) -> T:
    """Run one route's work, recording the request metrics either way."""

    start = time.perf_counter()
    outcome = "ok"
    try:
        return call()
    except ServingError as exc:
        outcome = exc.code
        raise
    finally:
        labels = {"endpoint": endpoint, "outcome": outcome}
        API_REQUESTS.add(1, labels)
        API_REQUEST_DURATION.record(time.perf_counter() - start, labels)


def _error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}, "trace_id": get_trace_id()}


def _version() -> str:
    from kuhaku import __version__

    return __version__


class _TraceIdMiddleware:
    """Binds a fresh trace id for the whole request and returns it as ``X-Trace-Id``.

    Plain ASGI rather than ``BaseHTTPMiddleware`` so the bound ``ContextVar`` is the one
    the route -- and, through it, ``RAGEngine.answer`` and ``Extractor.extract``, which
    inherit an outer id -- actually sees. The id is always minted here: a client-supplied
    id would let a caller forge or collide entries in the audit trail.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        with bind_trace_id(_fresh_trace_id()) as tid:

            async def send_with_trace_id(message: Any) -> None:
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((TRACE_ID_HEADER.lower().encode(), tid.encode()))
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_with_trace_id)


def _fresh_trace_id() -> str:
    from kuhaku.core.observability import new_trace_id

    return new_trace_id()
