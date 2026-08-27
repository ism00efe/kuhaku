"""RAG facade: ``retrieval="auto"`` (the shipped default) resolves to hybrid when an
embedding provider builds, and downgrades to sparse-only -- announced on the terminal --
when it does not."""

from __future__ import annotations

import warnings

import pytest

import kuhaku
from kuhaku import RAG
from kuhaku.core import capabilities as cap
from kuhaku.core.config import Settings
from kuhaku.core.exceptions import FallbackWarning
from kuhaku.tools.rag.embeddings import NullEmbeddings
from tests.conftest import FakeEmbeddings, FakeLLM, FakeVectorStore


@pytest.fixture(autouse=True)
def _clear_emitted():
    cap.reset_emitted()
    yield
    cap.reset_emitted()


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: FakeVectorStore())
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s: FakeLLM())

    def _build(**kwargs):
        kwargs.setdefault("cache", False)
        return RAG(settings=Settings(_env_file=None, audit_enabled=False), **kwargs)

    return _build, monkeypatch


def test_auto_resolves_to_hybrid_when_embedder_builds(patched):
    build, mp = patched
    mp.setattr(kuhaku, "build_embedding_provider", lambda rs: FakeEmbeddings())
    rag = build()
    assert rag.engine.get_retriever().strategy == "hybrid"


def test_auto_downgrades_to_sparse_when_embedder_unavailable(patched, capsys):
    build, mp = patched

    def _boom(rs):
        raise ImportError("No module named 'torch'")

    mp.setattr(kuhaku, "build_embedding_provider", _boom)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rag = build()

    assert rag.engine.get_retriever().strategy == "sparse"
    assert isinstance(rag._embedder, NullEmbeddings)
    assert any(isinstance(w.message, FallbackWarning) for w in caught)
    assert "retrieval" in capsys.readouterr().err


def test_explicit_hybrid_still_fails_loudly_when_embedder_unavailable(patched):
    build, mp = patched

    def _boom(rs):
        raise ImportError("No module named 'torch'")

    mp.setattr(kuhaku, "build_embedding_provider", _boom)
    with pytest.raises(ImportError):
        build(retrieval="hybrid")


def test_explicit_sparse_never_touches_the_embedder(patched):
    build, mp = patched

    def _boom(rs):
        raise AssertionError("build_embedding_provider must not be called for retrieval='sparse'")

    mp.setattr(kuhaku, "build_embedding_provider", _boom)
    rag = build(retrieval="sparse")
    assert isinstance(rag._embedder, NullEmbeddings)
    assert rag.engine.get_retriever().strategy == "sparse"


def test_auto_disabled_treats_auto_as_hybrid(patched):
    build, mp = patched
    mp.setenv("KUHAKU_AUTO", "false")

    def _boom(rs):
        raise ImportError("torch missing")

    mp.setattr(kuhaku, "build_embedding_provider", _boom)
    with pytest.raises(ImportError):
        build()


def test_sparse_only_ingest_and_query_roundtrip(patched):
    build, mp = patched

    def _boom(rs):
        raise ImportError("No module named 'sentence_transformers'")

    mp.setattr(kuhaku, "build_embedding_provider", _boom)
    rag = build()
    rag.ingest("Refund code PAY-9911 takes 1-5 business days.", "faq.md")
    results = rag.engine.retrieve("PAY-9911", top_k=5)
    assert any("PAY-9911" in r.chunk.text for r in results)
