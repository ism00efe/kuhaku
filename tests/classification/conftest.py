"""Fakes for the classification tests: an LLM that replays a script, and a label set."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from kuhaku.core.llm.base import LLMError, TokenUsage
from kuhaku.tools.classification import Label

Reply = str | dict[str, Any] | Exception | Callable[[str, str], str]


class ScriptedLLM:
    """Returns its scripted replies in order and records every prompt it was sent.

    A dict reply is sent as JSON, an exception is raised, a callable is called with
    ``(system, user)``. Running past the end of the script fails the test loudly rather
    than returning something plausible. With ``usage``, every call sets a fresh
    ``last_usage`` the way the real providers do.
    """

    def __init__(
        self, *replies: Reply, name: str = "scripted:model", usage: TokenUsage | None = None
    ) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []
        self._name = name
        self._usage = usage
        self.last_usage: TokenUsage | None = None

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
        if self._usage is not None:
            self.last_usage = TokenUsage(
                prompt_tokens=self._usage.prompt_tokens,
                completion_tokens=self._usage.completion_tokens,
            )
        if callable(reply):
            return reply(system, user)
        return reply if isinstance(reply, str) else json.dumps(reply)


def down(message: str = "connection refused") -> LLMError:
    return LLMError(message)


def answer(*pairs: tuple[str, float], reason: str = "because") -> dict[str, Any]:
    """A well-formed reply proposing ``(label, confidence)`` pairs."""

    return {
        "labels": [{"label": name, "confidence": confidence} for name, confidence in pairs],
        "reason": reason,
    }


LABELS = [
    Label("billing", "Invoices, charges and refunds", examples=("Why was I charged twice?",)),
    Label("technical", "Bugs, errors, crashes"),
    Label("account", "Login, password, profile"),
]
