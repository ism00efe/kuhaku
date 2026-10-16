"""Fakes for the extraction tests: an LLM that replays a script, and two small schemas."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from kuhaku.core.llm.base import LLMError

Reply = str | dict[str, Any] | Exception | Callable[[str, str], str]


class ScriptedLLM:
    """Returns its scripted replies in order and records every prompt it was sent.

    A dict reply is sent as JSON, an exception is raised, a callable is called with
    ``(system, user)``. Running past the end of the script fails the test loudly rather
    than returning something plausible.
    """

    def __init__(self, *replies: Reply, name: str = "scripted:model") -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise AssertionError(f"{self._name} was called more times than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(system, user)
        return reply if isinstance(reply, str) else json.dumps(reply)


def down(message: str = "connection refused") -> LLMError:
    return LLMError(message)


class Order(BaseModel):
    """A customer placing an order."""

    product: str
    quantity: float


class Complaint(BaseModel):
    """A customer complaining about something."""

    topic: str
