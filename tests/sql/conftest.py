"""Fakes and fixtures for the SQL tool tests: a scripted LLM, a native-structured-output
LLM, and a small SQLite database with something worth hiding in it."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

from kuhaku.core.auth import AuthContext
from kuhaku.core.config import Settings
from kuhaku.core.llm import (
    BaseLLMProvider,
    LLMCapabilities,
    LLMResponse,
    StructuredOutputMode,
    TokenUsage,
)
from kuhaku.tools.sql import AccessPolicy, RowFilter, SQLiteDatabase

Reply = str | dict[str, Any] | Exception | Callable[[str, str], str]


class ScriptedLLM:
    """A v1 provider replaying scripted replies and recording every prompt it was sent.
    Running past the end of the script fails the test loudly."""

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

    @property
    def prompts(self) -> str:
        return "\n".join(system + "\n" + user for system, user in self.calls)


class NativeLLM(BaseLLMProvider):
    """A v2 provider with a native schema mode that reports token usage."""

    capabilities = LLMCapabilities(chat=True, structured_output=StructuredOutputMode.SCHEMA)

    def __init__(self, *replies: Any) -> None:
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.native_calls: list[Any] = []
        self.chat_calls: list[Any] = []

    @property
    def name(self) -> str:
        return "native:model"

    def chat(self, messages, *, options=None) -> LLMResponse:
        self.chat_calls.append(tuple(messages))
        return LLMResponse(
            text=self.replies.pop(0),
            usage=TokenUsage(prompt_tokens=7, completion_tokens=3),
            model=self.name,
        )

    def _structured_complete(self, messages, request, options) -> LLMResponse:
        self.native_calls.append((messages, request))
        return LLMResponse(
            text=self.replies.pop(0),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            model=self.name,
        )


def sql(query: str | None, explanation: str = "because") -> dict[str, Any]:
    return {"sql": query, "explanation": explanation}


HR = AuthContext("hana", is_authenticated=True, roles=("hr",), metadata={"tenant": "acme"})
ENG = AuthContext("ada", is_authenticated=True, roles=("eng",), metadata={"tenant": "acme"})
GLOBEX = AuthContext("gus", is_authenticated=True, roles=("eng",), metadata={"tenant": "globex"})

# Names that exist only where the caller may not look. A prompt or a message the model
# sees must never contain any of them.
HIDDEN_NAMES = ("salaries", "amount", "salary", "email")


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "shop.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE depts (id INTEGER PRIMARY KEY, title TEXT);
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, email TEXT,
            salary INTEGER, tenant TEXT
        );
        CREATE TABLE salaries (emp_id INTEGER, amount INTEGER);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, tenant TEXT, customer TEXT, total REAL);
        INSERT INTO depts VALUES (1, 'Engineering'), (2, 'People');
        INSERT INTO employees VALUES
            (1, 'Ada', 1, 'ada@acme.test', 120, 'acme'),
            (2, 'Grace', 1, 'grace@acme.test', 130, 'acme'),
            (3, 'Hana', 2, 'hana@acme.test', 110, 'acme'),
            (4, 'Gus', 1, 'gus@globex.test', 90, 'globex');
        INSERT INTO salaries VALUES (1, 120), (2, 130), (3, 110), (4, 90);
        INSERT INTO orders VALUES
            (1, 'acme', 'Initech', 100.0), (2, 'acme', 'Umbrella', 250.0),
            (3, 'globex', 'Hooli', 999.0);
        """
    )
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def database(db_path):
    return SQLiteDatabase(db_path)


def tenant_filter(auth: AuthContext | None) -> RowFilter:
    # Fail closed: no AuthContext means a NULL tenant, which matches no row.
    return RowFilter(
        "tenant = :tenant", {"tenant": auth.metadata.get("tenant") if auth else None}
    )


@pytest.fixture
def policy():
    return AccessPolicy(
        tables={"salaries": ["hr"]},
        columns={"employees.salary": ["hr"], "employees.email": ["hr"]},
        row_filters={"orders": tenant_filter, "employees": tenant_filter},
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(audit_log_path=str(tmp_path / "audit.jsonl"))
