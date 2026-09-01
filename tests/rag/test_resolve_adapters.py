"""RAG resolution adapters: embedding (local + API) and store, plus the embedding-axis
wiring in RAG() (§7). Replaces the old tests/rag/test_capabilities.py."""

from __future__ import annotations

import pytest

from kuhaku.core.resolve import Registry
from kuhaku.core.resolve.environment import Environment
from kuhaku.tools.rag.config import RAGSettings
from kuhaku.tools.rag.embeddings import build_embedding_provider
from kuhaku.tools.rag.resolve import build_rag_registry
from kuhaku.tools.rag.resolve.adapters import embedding as embedding_adapters


def _env() -> Environment:
    return Environment(
        python="3.12", in_isolated_python=True, isolation_source="venv",
        gpu=None, vram_class="none", os="linux",
    )


def _rag_registry(rs, *, embedder=object) -> Registry:
    return build_rag_registry(rs, build_embedder=lambda: embedder, build_store=lambda: object())


# --- local embedding adapter -------------------------------------------------
def test_local_embedding_ready_only_when_package_and_model_present(monkeypatch):
    monkeypatch.setattr(embedding_adapters, "module_available", lambda name: True)
    monkeypatch.setattr(embedding_adapters, "_model_cached", lambda name: True)
    reg = _rag_registry(RAGSettings())
    (cand,) = [c for c in reg.candidates("embedding", _env()) if c.id == "sentence-transformer"]
    assert cand.ready is True
    assert cand.cost.sends_document_text is False  # local -> nothing leaves the machine


def test_local_embedding_not_ready_without_model(monkeypatch):
    monkeypatch.setattr(embedding_adapters, "module_available", lambda name: True)
    monkeypatch.setattr(embedding_adapters, "_model_cached", lambda name: False)
    reg = _rag_registry(RAGSettings())
    (cand,) = [c for c in reg.candidates("embedding", _env()) if c.id == "sentence-transformer"]
    assert cand.ready is False
    assert cand.cost.download_required is True


# --- API embedding adapter: §7 -- an LLM key never routes here --------------
def test_api_embedding_absent_unless_explicitly_chosen():
    reg = _rag_registry(RAGSettings(vertex_project="p"))  # project set, but provider not "vertex"
    ids = {c.id for c in reg.candidates("embedding", _env())}
    assert "vertex-embeddings" not in ids


def test_api_embedding_present_when_explicitly_chosen():
    rs = RAGSettings(embedding_provider="vertex", vertex_project="p")
    reg = _rag_registry(rs)
    (cand,) = [c for c in reg.candidates("embedding", _env()) if c.id == "vertex-embeddings"]
    assert cand.ready is True
    assert cand.cost.sends_document_text is True  # ingest-time document text leaves the machine


# --- store adapter ---------------------------------------------------------
def test_store_adapter_offers_chroma():
    reg = _rag_registry(RAGSettings())
    ids = {c.id for c in reg.candidates("store", _env())}
    assert ids == {"chroma"}


# --- build_embedding_provider device handling ------------------------------
def test_build_embedding_provider_uses_passed_device(monkeypatch):
    captured = {}

    class _Fake:
        def __init__(self, name, *, device=None, **kw):
            captured["device"] = device

        def embed_documents(self, t):  # pragma: no cover
            return []

        def embed_query(self, t):  # pragma: no cover
            return []

    monkeypatch.setattr(
        "kuhaku.tools.rag.embeddings.SentenceTransformerEmbeddings", _Fake
    )
    build_embedding_provider(RAGSettings(embedding_device="auto"), device="cuda")
    assert captured["device"] == "cuda"
