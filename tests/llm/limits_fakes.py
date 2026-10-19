"""Shared fakes for the rate-limit and budget tests: a clock that never really sleeps,
a scriptable provider, and a 429 shaped like the one ``requests`` raises."""

from __future__ import annotations

import threading
from collections.abc import Callable

import requests

from kuhaku.core.llm.base import LLMError, TokenUsage


class FakeClock:
    """Monotonic clock whose ``sleep`` advances time instantly and records the call."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class FakeProvider:
    """LLMProvider whose replies, usage reports and failures are scripted per call.

    Each entry of ``script`` is either a ``(text, TokenUsage | None)`` pair or an
    exception to raise. A ``None`` usage leaves ``last_usage`` untouched, the way the
    real providers behave when a response carries no usage.
    """

    def __init__(
        self,
        name: str = "fake:model-x",
        script: list[object] | None = None,
        *,
        default: tuple[str, TokenUsage | None] = ("ok", None),
    ) -> None:
        self._name = name
        self.script = list(script or [])
        self.default = default
        self.calls: list[tuple[str, str]] = []
        self.last_usage: TokenUsage | None = None
        self.on_call: Callable[[], None] | None = None

    @property
    def name(self) -> str:
        return self._name

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.on_call is not None:
            self.on_call()
        step = self.script.pop(0) if self.script else self.default
        if isinstance(step, BaseException):
            raise step
        text, usage = step  # type: ignore[misc]
        if usage is not None:
            self.last_usage = usage
        return text


def http_error(status: int, headers: dict[str, str] | None = None) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    return requests.HTTPError(f"{status} error", response=response)


def wrapped_http_error(status: int, headers: dict[str, str] | None = None) -> LLMError:
    """An ``LLMError`` raised ``from`` an HTTP error -- the shape every built-in provider
    produces."""

    try:
        try:
            raise http_error(status, headers)
        except requests.HTTPError as exc:
            raise LLMError(f"request failed: {exc}") from exc
    except LLMError as wrapped:
        return wrapped
