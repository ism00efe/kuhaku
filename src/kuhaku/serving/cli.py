"""``kuhaku-mcp``: run a :class:`~kuhaku.serving.mcp.KuhakuMCPServer` configured in Python.

    kuhaku-mcp --config my_server:server
    kuhaku-mcp --config ./my_server.py:build --transport streamable-http --port 8000

``--config`` names a ``module:attribute`` (or ``path/to/file.py:attribute``). The
attribute is a ``KuhakuMCPServer``, or a zero-argument callable returning one. All
configuration -- the RAG, the extractors, the identity -- lives in that Python file;
the CLI only picks the transport.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from kuhaku.core.exceptions import ConfigError

from .mcp import TRANSPORTS, KuhakuMCPServer


def load_server(spec: str) -> KuhakuMCPServer:
    """Resolve a ``module:attribute`` / ``file.py:attribute`` spec to a server."""

    target, sep, attribute = spec.rpartition(":")
    # "C:\\x.py" has a drive colon, not an attribute separator.
    if not sep or not attribute or not target or (len(target) == 1 and target.isalpha()):
        raise ConfigError(f"--config must look like 'module:attribute', got {spec!r}.")
    module = _load_module(target)
    try:
        obj: Any = getattr(module, attribute)
    except AttributeError:
        raise ConfigError(f"{target!r} has no attribute {attribute!r}.") from None
    if not isinstance(obj, KuhakuMCPServer) and callable(obj):
        obj = obj()
    if not isinstance(obj, KuhakuMCPServer):
        raise ConfigError(
            f"{spec!r} must be a KuhakuMCPServer or a callable returning one, "
            f"got {type(obj).__name__}."
        )
    return obj


def _load_module(target: str) -> Any:
    if target.endswith(".py"):
        path = Path(target).resolve()
        if not path.is_file():
            raise ConfigError(f"No such file: {path}")
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ConfigError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[path.stem] = module
        module_spec.loader.exec_module(module)
        return module
    # A module next to where the command runs should be importable, as with `python -m`.
    if "" not in sys.path and str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kuhaku-mcp", description="Serve kuhaku tools over MCP."
    )
    parser.add_argument("--config", required=True, help="module:attribute or file.py:attribute")
    parser.add_argument("--transport", choices=TRANSPORTS, default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/mcp", help="URL path for streamable HTTP")
    args = parser.parse_args(argv)

    try:
        if args.transport == "stdio":
            # Building the server can print (RAG.load_documents does); over stdio, stdout
            # belongs to the protocol, so anything printed while loading goes to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                server = load_server(args.config)
        else:
            server = load_server(args.config)
        server.run(args.transport, host=args.host, port=args.port, path=args.path)
    except ConfigError as exc:
        print(f"kuhaku-mcp: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
