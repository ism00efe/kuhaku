"""Request-size limits shared by both transports.

The wire schemas cap each text field (``max_length``), but pydantic only sees a body
after it has been read and parsed in full; :class:`BodySizeLimit` refuses an oversized
body before that, at the ASGI layer, so a multi-gigabyte POST costs the server at most
``max_body_bytes`` of memory.
"""

from __future__ import annotations

import json
from typing import Any

# Comfortably above the largest request the field limits allow (an ingested document of
# MAX_INGEST_TEXT_CHARS, JSON-escaped), far below anything that hurts a server.
DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024

MAX_QUESTION_CHARS = 8_000
MAX_CONTEXT_TEXT_CHARS = 100_000
MAX_EXTRACT_TEXT_CHARS = 100_000
MAX_INGEST_TEXT_CHARS = 2_000_000


class _TooLarge(Exception):
    pass


class BodySizeLimit:
    """ASGI middleware: answer 413 to any HTTP request whose body exceeds ``max_bytes``.

    A declared ``Content-Length`` over the limit is refused without reading anything.
    Otherwise the body is read (at most ``max_bytes`` + one chunk) before the wrapped
    app runs, so a chunked upload without a length is bounded the same way.
    """

    def __init__(self, app: Any, max_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("max_body_bytes must be >= 1")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = -1
                if declared < 0 or declared > self.max_bytes:
                    await _send_413(send, self.max_bytes)
                    return
        try:
            messages = await self._read_body(receive)
        except _TooLarge:
            await _send_413(send, self.max_bytes)
            return

        async def replay() -> Any:
            if messages:
                return messages.pop(0)
            # After the body, the app may wait for http.disconnect (SSE streams do).
            return await receive()

        await self.app(scope, replay, send)

    async def _read_body(self, receive: Any) -> list[Any]:
        messages: list[Any] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                return messages
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                raise _TooLarge
            if not message.get("more_body", False):
                return messages


async def _send_413(send: Any, limit: int) -> None:
    body = json.dumps(
        {
            "error": {
                "code": "payload_too_large",
                "message": f"The request body exceeds {limit} bytes.",
            }
        }
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
