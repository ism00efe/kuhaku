"""RAG facade: ``retrieval="auto"`` resolves to hybrid when a real embedding backend is
usable now (package installed AND model already on disk), and degrades to sparse -- with
a FallbackWarning, and without downloading anything -- when it is not (§7)."""

from __future__ import annotations

import warnings

import pytest

import kuhaku
from kuhaku import RAG
from kuhaku.core.config import Settings
from kuhaku.core.exceptions import CapabilityUnavailable, ConsentRequired, FallbackWarning
from kuhaku.tools.rag.embeddings import NullEmbeddings
from tests.conftest import FakeEmbeddings, FakeLLM, FakeVectorStore


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: FakeVectorStore())
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s, **k: FakeLLM())

    def _model_present(present: bool):
        monkeypatch.setattr(
            "kuhaku.tools.rag.resolve.adapters.embedding._model_cached",
            lambda name: present,
        )

    def _build(**kwargs):
        kwargs.setdefault("cache", False)
        return RAG(settings=Settings(_env_file=None, audit_enabled=False), **kwargs)

    return _build, monkeypatch, _model_present


def test_auto_resolves_to_hybrid_when_a_real_embedder_is_ready(patched):
    build, mp, model_present = patched
    model_present(True)
    mp.setattr(kuhaku, "build_embedding_provider", lambda rs, **k: FakeEmbeddings())
    rag = build()
    assert rag.engine.get_retriever().strategy == "hybrid"


def test_auto_degrades_to_sparse_when_model_absent_without_downloading(patched):
    build, mp, model_present = patched
    model_present(False)

    def _must_not_build(rs, **k):
        raise AssertionError("auto must not build (download) an embedder when the model is absent")

    mp.setattr(kuhaku, "build_embedding_provider", _must_not_build)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rag = build()

    assert rag.engine.get_retriever().strategy == "sparse"
    assert isinstance(rag._embedder, NullEmbeddings)
    assert any(isinstance(w.message, FallbackWarning) for w in caught)


def test_explicit_hybrid_without_a_ready_embedder_raises_consent_required(patched):
    build, mp, model_present = patched
    model_present(False)  # not ready -> consent needed -> non-interactive -> ConsentRequired
    mp.setattr(kuhaku, "build_embedding_provider", lambda rs, **k: FakeEmbeddings())
    with pytest.raises(ConsentRequired):
        build(retrieval="hybrid")


def test_explicit_sparse_never_touches_the_embedder(patched):
    build, mp, model_present = patched

    def _boom(rs, **k):
        raise AssertionError("build_embedding_provider must not be called for retrieval='sparse'")

    mp.setattr(kuhaku, "build_embedding_provider", _boom)
    rag = build(retrieval="sparse")
    assert isinstance(rag._embedder, NullEmbeddings)
    assert rag.engine.get_retriever().strategy == "sparse"


def test_auto_disabled_uses_hybrid_baseline_and_raises_if_unusable(patched):
    build, mp, model_present = patched
    mp.setenv("KUHAKU_AUTO", "false")

    def _boom(rs, **k):
        raise ImportError("torch missing")

    mp.setattr(kuhaku, "build_embedding_provider", _boom)
    with pytest.raises(CapabilityUnavailable):
        build()


def test_sparse_only_ingest_and_query_roundtrip(patched):
    build, mp, model_present = patched
    model_present(False)
    mp.setattr(kuhaku, "build_embedding_provider",
               lambda rs, **k: (_ for _ in ()).throw(AssertionError("no embedder in sparse mode")))
    rag = build()
    rag.ingest("Refund code PAY-9911 takes 1-5 business days.", "faq.md")
    results = rag.engine.retrieve("PAY-9911", top_k=5)
    assert any("PAY-9911" in r.chunk.text for r in results)
