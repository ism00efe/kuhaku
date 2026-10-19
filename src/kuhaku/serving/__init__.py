"""Serving: kuhaku's tools over MCP and HTTP.

A layer beside ``tools/``: it may import ``kuhaku.core`` and any tool; neither ever
imports it. The transports are optional extras and are imported lazily --
``kuhaku.serving`` itself needs neither installed:

- ``create_app`` -- the HTTP API (``pip install 'kuhaku[http]'``)
- ``KuhakuMCPServer`` / ``serve_mcp`` -- the MCP server (``pip install 'kuhaku[mcp]'``)

kuhaku still authenticates no one: :class:`RequestAuthenticator` hands a request's
credential to a configured :class:`~kuhaku.core.auth.AuthProvider` and passes the
resulting :class:`~kuhaku.core.auth.AuthContext` to every call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .auth import RequestAuthenticator
from .errors import (
    AuthenticationRequired,
    Conflict,
    InvalidRequest,
    NotFound,
    PermissionDenied,
    ServiceUnavailable,
    ServingError,
)
from .service import ToolService, extractor_resource

if TYPE_CHECKING:
    from .http import create_app
    from .mcp import KuhakuMCPServer, serve_mcp

_LAZY = {
    "create_app": "http",
    "KuhakuMCPServer": "mcp",
    "serve_mcp": "mcp",
}


def __getattr__(name: str) -> Any:
    # The transport modules are cheap to import (their SDKs are imported inside them on
    # first use), but keeping them out of `import kuhaku.serving` means an error in one
    # transport can never break the other.
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


__all__ = [
    "AuthenticationRequired",
    "Conflict",
    "InvalidRequest",
    "KuhakuMCPServer",
    "NotFound",
    "PermissionDenied",
    "RequestAuthenticator",
    "ServiceUnavailable",
    "ServingError",
    "ToolService",
    "create_app",
    "extractor_resource",
    "serve_mcp",
]
