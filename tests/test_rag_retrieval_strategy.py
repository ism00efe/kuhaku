"""Tests for `RAG._build_retriever`'s routing between dense/sparse/hybrid strategies.

Notably: `retrieval="sparse"` routes through `SparseRetriever`, never `HybridRetriever`
(which is reserved for genuine dense+sparse fusion), so BM25's raw, unbounded Okapi
scores get min-max normalized before reaching the confidence threshold -- see
`kuhaku/tools/rag/retriever.py`'s `SparseRetriever.retrieve`.
"""

from __future__ import annotations

from types import SimpleNamespace

import kuhaku
from kuhaku import RAG
from kuhaku.core.config import Settings
from kuhaku.tools.rag.config import RAGSettings
from kuhaku.tools.rag.models import RetrievedChunk
from kuhaku.tools.rag.retriever import DenseRetriever, HybridRetriever, SparseRetriever
from tests.conftest import FakeEmbeddings, FakeVectorStore, make_chunk


class _RawScoreRetriever:
    """A minimal `Retriever` returning fixed, caller-supplied (possibly unbounded)
    scores -- stands in for BM25 without needing a real corpus on disk."""

    def __init__(self, items: list[RetrievedChunk]) -> None:
        self._items = items

    def retrieve(self, query, top_k, *, auth_context=None, doc_type=None, enforce_entitlement=True):
        return self._items[:top_k]


def _stub_rag(**overrides) -> SimpleNamespace:
    """A bare object with just the attributes `_build_retriever` reads -- avoids
    constructing a full `RAG()` (real embedding model, Chroma store, etc.)."""

    settings = Settings(_env_file=None, **overrides)
    return SimpleNamespace(
        _settings=settings,
        _rag_settings=RAGSettings.from_settings(settings),
        _embedder=FakeEmbeddings(),
        _store=FakeVectorStore(),
        _chunker=None,
    )


def test_build_retriever_dense_returns_bare_dense_retriever():
    stub = _stub_rag()
    retriever = RAG._build_retriever(stub, "dense", False)
    assert isinstance(retriever, DenseRetriever)


def test_build_retriever_sparse_without_reranker_routes_through_sparse_retriever(monkeypatch):
    fake_sparse = _RawScoreRetriever([RetrievedChunk(chunk=make_chunk("a"), score=8.3)])
    monkeypatch.setattr(kuhaku, "build_bm25_from_store", lambda *a, **k: fake_sparse)

    stub = _stub_rag()
    retriever = RAG._build_retriever(stub, "sparse", False)

    assert isinstance(retriever, SparseRetriever)
    assert not isinstance(retriever, HybridRetriever)
    assert retriever.strategy == "sparse"


def test_build_retriever_sparse_normalizes_raw_bm25_scores(monkeypatch):
    # Raw Okapi scores are unbounded (not 0..1) -- here 8.3/2.1/0.4 -- and must be
    # min-max normalized once routed through the sparse strategy with no reranker.
    fake_sparse = _RawScoreRetriever(
        [
            RetrievedChunk(chunk=make_chunk("hi"), score=8.3),
            RetrievedChunk(chunk=make_chunk("mid", 1), score=2.1),
            RetrievedChunk(chunk=make_chunk("lo", 2), score=0.4),
        ]
    )
    monkeypatch.setattr(kuhaku, "build_bm25_from_store", lambda *a, **k: fake_sparse)

    stub = _stub_rag()
    retriever = RAG._build_retriever(stub, "sparse", False)
    results = retriever.retrieve("query", 3)

    scores = {r.chunk.document_id: r.score for r in results}
    assert scores["hi"] == 1.0
    assert scores["lo"] == 0.0
    assert 0.0 < scores["mid"] < 1.0
    assert [r.chunk.document_id for r in results] == ["hi", "mid", "lo"]


def test_build_retriever_sparse_single_chunk_does_not_divide_by_zero(monkeypatch):
    fake_sparse = _RawScoreRetriever([RetrievedChunk(chunk=make_chunk("only"), score=5.0)])
    monkeypatch.setattr(kuhaku, "build_bm25_from_store", lambda *a, **k: fake_sparse)

    stub = _stub_rag()
    retriever = RAG._build_retriever(stub, "sparse", False)
    results = retriever.retrieve("query", 1)

    assert results[0].score == 1.0


def test_build_retriever_sparse_with_reranker_strategy(monkeypatch):
    fake_sparse = _RawScoreRetriever([RetrievedChunk(chunk=make_chunk("a"), score=8.3)])
    monkeypatch.setattr(kuhaku, "build_bm25_from_store", lambda *a, **k: fake_sparse)

    stub = _stub_rag()
    retriever = RAG._build_retriever(stub, "sparse", True)

    assert isinstance(retriever, SparseRetriever)
    assert retriever.strategy == "sparse+rerank"


def test_build_retriever_hybrid_uses_hybrid_retriever(monkeypatch):
    fake_sparse = _RawScoreRetriever([RetrievedChunk(chunk=make_chunk("a"), score=8.3)])
    monkeypatch.setattr(kuhaku, "build_bm25_from_store", lambda *a, **k: fake_sparse)

    stub = _stub_rag()
    retriever = RAG._build_retriever(stub, "hybrid", False)

    assert isinstance(retriever, HybridRetriever)
    assert retriever.strategy == "hybrid"


def test_build_retriever_dense_with_reranker_strategy():
    stub = _stub_rag()
    retriever = RAG._build_retriever(stub, "dense", True)

    assert isinstance(retriever, HybridRetriever)
    assert retriever.strategy == "dense+rerank"
