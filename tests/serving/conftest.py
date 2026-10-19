"""Fixtures for the serving tests: a real ``kuhaku.RAG`` on the in-memory fakes, with the
audit log on (into tmp_path) so tests can check who the trail says asked, and an
``Extractor`` on a scripted LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

import kuhaku
from kuhaku import RAG, Extractor
from kuhaku.core.auth import APIKeyAuthProvider
from kuhaku.core.config import Settings
from tests.conftest import FakeEmbeddings, FakeLLM, FakeVectorStore
from tests.extraction.conftest import ScriptedLLM

SALARY = "# Salary Policy\n\nSalary bands range from A to E based on level."
DEPLOY = "# Deploy Runbook\n\nRun deploy.sh to ship a new release."

ADMIN_KEY = "key-admin"
HR_KEY = "key-hr"
ENG_KEY = "key-eng"


class Order(BaseModel):
    """A customer placing an order."""

    product: str
    quantity: float


@dataclass
class Served:
    rag: RAG
    llm: FakeLLM
    audit_path: Path

    def audit_records(self) -> list[dict]:
        if not self.audit_path.exists():
            return []
        return [json.loads(line) for line in self.audit_path.read_text("utf-8").splitlines()]


@pytest.fixture
def served(monkeypatch, tmp_path) -> Served:
    store, llm = FakeVectorStore(), FakeLLM()
    monkeypatch.setattr(kuhaku, "build_embedding_provider", lambda rs: FakeEmbeddings())
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: store)
    monkeypatch.setattr(
        "kuhaku.tools.rag.resolve.adapters.embedding._model_cached", lambda name: True
    )
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s, **_k: llm)
    audit_path = tmp_path / "audit.jsonl"
    settings = Settings(_env_file=None, audit_enabled=True, audit_log_path=str(audit_path))
    rag = RAG(
        settings=settings,
        cache=False,
        retrieval="dense",
        vector_store=str(tmp_path / "store"),
    )
    rag.ingest(SALARY, filename="salary_policy.md", access_tags=["people_ops"])
    rag.ingest(DEPLOY, filename="deploy_runbook.md")
    return Served(rag=rag, llm=llm, audit_path=audit_path)


@pytest.fixture
def keys() -> APIKeyAuthProvider:
    return APIKeyAuthProvider(
        {
            ADMIN_KEY: {"identity": "root", "roles": ["admin"]},
            HR_KEY: {"identity": "hana", "roles": ["people_ops"]},
            ENG_KEY: {"identity": "efe", "roles": ["engineering"]},
        }
    )


def order_extractor(*replies) -> Extractor:
    llm = ScriptedLLM(*replies)
    return Extractor(Order, llm=llm, enable_token_tracking=False)
