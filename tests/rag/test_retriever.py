"""Tests for the retriever abstraction, RRF fusion, and the hybrid chain."""

from __future__ import annotations

import pytest

from kuhaku.tools.rag.models import RetrievedChunk
from kuhaku.tools.rag.embeddings import EmbeddingServiceError
from kuhaku.tools.rag.retriever import (
    DenseRetriever,
    HybridRetriever,
    SparseRetriever,
    _min_max_normalize,
    reciprocal_rank_fusion,
)
from kuhaku.tools.rag.vectorstore import VectorStoreError
from tests.conftest import FakeEmbeddings, FakeReranker, FakeRetriever, FakeVectorStore, make_chunk


def _ranked(*doc_ids: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(make_chunk(d), 1.0) for d in doc_ids]


# --- DenseRetriever adapter -------------------------------------------------
def test_dense_retriever_embeds_then_queries():
    embedder = FakeEmbeddings()
    store = FakeVectorStore([make_chunk("a"), make_chunk("b", 1)])
    results = DenseRetriever(embedder, store).retrieve("soru", 2)

    assert embedder.query_calls == ["soru"]
    assert [r.chunk.document_id for r in results] == ["a", "b"]


def test_dense_retriever_top_k_zero():
    assert DenseRetriever(FakeEmbeddings(), FakeVectorStore()).retrieve("q", 0) == []


# --- FR2: graceful degradation -----------------------------------------------
class _RaisingEmbedder:
    def embed_documents(self, texts):  # pragma: no cover - not exercised here
        raise RuntimeError("model not loaded")

    def embed_query(self, text):
        raise RuntimeError("model not loaded")


class _RaisingStore:
    def add(self, chunks, embeddings):  # pragma: no cover - not exercised here
        raise RuntimeError("connection refused")

    def query(self, embedding, top_k):
        raise RuntimeError("connection refused")

    def count(self):  # pragma: no cover - not exercised here
        raise RuntimeError("connection refused")

    def reset(self):  # pragma: no cover - not exercised here
        raise RuntimeError("connection refused")


def test_dense_retriever_wraps_embedding_failures():
    retriever = DenseRetriever(_RaisingEmbedder(), FakeVectorStore())
    with pytest.raises(EmbeddingServiceError):
        retriever.retrieve("q", 2)


def test_dense_retriever_wraps_vector_store_failures():
    retriever = DenseRetriever(FakeEmbeddings(), _RaisingStore())
    with pytest.raises(VectorStoreError):
        retriever.retrieve("q", 2)


# --- FR4: freshness filtering ---------------------------------------------------
def test_dense_retriever_filters_out_obsolete_chunks():
    store = FakeVectorStore([
        make_chunk("fresh"),
        make_chunk("stale", 1, obsolete=True),
    ])
    results = DenseRetriever(FakeEmbeddings(), store).retrieve("q", 2)
    assert [r.chunk.document_id for r in results] == ["fresh"]


def test_dense_retriever_filters_out_expired_chunks():
    store = FakeVectorStore([
        make_chunk("fresh"),
        make_chunk("expired", 1, expiry_date="2020-01-01"),
    ])
    results = DenseRetriever(FakeEmbeddings(), store).retrieve("q", 2)
    assert [r.chunk.document_id for r in results] == ["fresh"]


def test_dense_retriever_updates_active_documents_gauge():
    from tests.conftest import prometheus_gauge_value
    from kuhaku.tools.rag.metrics import ACTIVE_DOCUMENTS

    store = FakeVectorStore([make_chunk("a"), make_chunk("b", 1, obsolete=True)])
    DenseRetriever(FakeEmbeddings(), store).retrieve("q", 2)
    assert prometheus_gauge_value(ACTIVE_DOCUMENTS) == 1


# --- auth_context pass-through: built-in retrievers filter nothing based on it -----
def test_dense_retriever_performs_no_filtering_based_on_auth_context():
    """The built-in retrievers do not interpret auth_context at all -- authorization
    decisions live in RAGEngine's optional AuthorizationPolicy (kuhaku.core.auth), not
    in the retriever layer. Passing any AuthContext must not change the result."""

    from kuhaku.core.auth import AuthContext

    store = FakeVectorStore([make_chunk("a"), make_chunk("b", 1)])
    unauth = DenseRetriever(FakeEmbeddings(), store).retrieve("q", 4)
    authed = DenseRetriever(FakeEmbeddings(), store).retrieve(
        "q", 4, auth_context=AuthContext(identity="u1", is_authenticated=True)
    )
    assert [r.chunk.document_id for r in unauth] == [r.chunk.document_id for r in authed]


def test_dense_retriever_doc_type_filter_excludes_other_types():
    store = FakeVectorStore(
        [make_chunk("faq1", doc_type="faq"), make_chunk("runbook1", doc_type="runbook")]
    )
    results = DenseRetriever(FakeEmbeddings(), store).retrieve("q", 4, doc_type="runbook")
    assert [r.chunk.document_id for r in results] == ["runbook1"]


def test_dense_retriever_doc_type_none_is_unfiltered():
    store = FakeVectorStore(
        [make_chunk("faq1", doc_type="faq"), make_chunk("runbook1", doc_type="runbook")]
    )
    results = DenseRetriever(FakeEmbeddings(), store).retrieve("q", 4, doc_type=None)
    assert {r.chunk.document_id for r in results} == {"faq1", "runbook1"}


# --- Reciprocal Rank Fusion -------------------------------------------------
def test_rrf_rewards_agreement_across_rankings():
    """A document ranked well by BOTH retrievers beats one ranked first by only one."""
    dense = _ranked("only_dense", "agreed")
    sparse = _ranked("only_sparse", "agreed")

    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert fused[0].chunk.document_id == "agreed"


def test_rrf_scores_match_hand_computation():
    dense = _ranked("x", "y")
    sparse = _ranked("y")
    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    scores = {r.chunk.document_id: r.score for r in fused}
    # x: only dense at rank 1 -> 1/61 ; y: dense rank 2 + sparse rank 1 -> 1/62 + 1/61
    assert scores["x"] == 1 / 61
    assert scores["y"] == 1 / 62 + 1 / 61
    assert fused[0].chunk.document_id == "y"  # higher fused score wins


def test_rrf_k_dampens_rank_differences():
    dense = _ranked("first", "second")
    small_k = reciprocal_rank_fusion([dense], k=1)
    large_k = reciprocal_rank_fusion([dense], k=1000)
    # A large k flattens the gap between rank 1 and rank 2.
    small_gap = small_k[0].score - small_k[1].score
    large_gap = large_k[0].score - large_k[1].score
    assert large_gap < small_gap


def test_rrf_deduplicates_chunks():
    fused = reciprocal_rank_fusion([_ranked("a"), _ranked("a")], k=60)
    assert len(fused) == 1
    assert fused[0].score == 2 * (1 / 61)


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_rrf_is_deterministic_on_ties():
    # Identical rankings in both lists -> equal scores -> tie broken by chunk id.
    fused = reciprocal_rank_fusion([_ranked("b", "a"), _ranked("a", "b")], k=60)
    assert [r.chunk.id for r in fused] == sorted(r.chunk.id for r in fused)


# --- min-max normalization helper (hybrid confidence scoring) --------------
def test_min_max_normalize_basic_range():
    assert _min_max_normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_min_max_normalize_single_value_positive_maps_to_one():
    assert _min_max_normalize([3.0]) == [1.0]


def test_min_max_normalize_single_value_zero_maps_to_zero():
    assert _min_max_normalize([0.0]) == [0.0]


def test_min_max_normalize_tie_at_nonzero_maps_to_one():
    assert _min_max_normalize([2.0, 2.0, 2.0]) == [1.0, 1.0, 1.0]


def test_min_max_normalize_tie_at_zero_maps_to_zero():
    assert _min_max_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_min_max_normalize_empty():
    assert _min_max_normalize([]) == []


# --- HybridRetriever --------------------------------------------------------
def test_hybrid_without_sparse_behaves_like_dense():
    dense = FakeRetriever([make_chunk("a"), make_chunk("b", 1)])
    hybrid = HybridRetriever(dense, sparse=None, reranker=None, candidates=20)
    results = hybrid.retrieve("q", 2)
    assert [r.chunk.document_id for r in results] == ["a", "b"]
    # Scores are untouched when there is nothing to fuse.
    assert results[0].score == 1.0


def test_hybrid_fuses_both_sources():
    dense = FakeRetriever([make_chunk("dense_only"), make_chunk("shared", 1)])
    sparse = FakeRetriever([make_chunk("shared", 1), make_chunk("sparse_only")])
    hybrid = HybridRetriever(dense, sparse, candidates=20)

    results = hybrid.retrieve("q", 3)
    ids = [r.chunk.document_id for r in results]
    assert ids[0] == "shared"  # agreed-upon document ranks first
    assert set(ids) == {"shared", "dense_only", "sparse_only"}


def test_hybrid_replaces_rrf_score_with_normalized_combined_score():
    """The final score must be a meaningful 0.5/0.5 dense+BM25 combination, not the
    tiny (~0.03) raw RRF fusion score -- that was the bug this whole feature fixes."""
    dense = FakeRetriever([make_chunk("a"), make_chunk("b", 1)])
    sparse = FakeRetriever([make_chunk("b", 1), make_chunk("a")])
    hybrid = HybridRetriever(dense, sparse, candidates=20)

    results = hybrid.retrieve("q", 2)
    scores = {r.chunk.id: r.score for r in results}
    # Symmetric setup (each doc ranks #1 on one side, #2 on the other) -> both land
    # exactly on the midpoint of the weighted, min-max-normalized combination.
    assert scores["a::0"] == pytest.approx(0.5)
    assert scores["b::1"] == pytest.approx(0.5)
    assert all(s > 0.15 for s in scores.values())  # clears the default confidence threshold


def test_hybrid_bm25_only_match_still_contributes_to_final_score():
    """A BM25 exact match with no dense counterpart must still pull the combined score
    up via its 0.5 weight -- not be discarded, the way a near-zero RRF score would be."""
    dense = FakeRetriever([make_chunk("other")])
    sparse = FakeRetriever([make_chunk("exact_match")])
    hybrid = HybridRetriever(dense, sparse, candidates=20)

    results = hybrid.retrieve("q", 2)
    scores = {r.chunk.id: r.score for r in results}
    assert scores["exact_match::0"] == pytest.approx(0.5)


class _ScoringReranker:
    """Assigns explicit per-chunk scores, simulating a real cross-encoder's output --
    unlike ``FakeReranker``, which only reorders and leaves incoming scores untouched."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def rerank(self, query, candidates, top_k):
        rescored = [
            RetrievedChunk(chunk=item.chunk, score=self._scores.get(item.chunk.id, 0.0))
            for item in candidates
        ]
        rescored.sort(key=lambda r: (-r.score, r.chunk.id))
        return rescored[:top_k]


def test_hybrid_with_reranker_uses_reranker_score_not_rrf_or_combined_score():
    dense = FakeRetriever([make_chunk("a"), make_chunk("b", 1)])
    sparse = FakeRetriever([make_chunk("b", 1), make_chunk("a")])
    reranker = _ScoringReranker({"a::0": 0.92, "b::1": 0.31})
    hybrid = HybridRetriever(dense, sparse, reranker=reranker, candidates=20)

    results = hybrid.retrieve("q", 2)
    scores = {r.chunk.id: r.score for r in results}
    assert scores == {"a::0": 0.92, "b::1": 0.31}


def test_hybrid_falls_back_to_combined_score_when_reranker_fails():
    """On reranker failure the fallback must be the normalized combined score, not the
    raw RRF fusion score it replaces -- same guarantee as the no-reranker path."""
    dense = FakeRetriever([make_chunk("a"), make_chunk("b", 1)])
    sparse = FakeRetriever([make_chunk("b", 1), make_chunk("a")])
    hybrid = HybridRetriever(dense, sparse, reranker=_RaisingReranker(), candidates=20)

    results = hybrid.retrieve("q", 2)
    scores = {r.chunk.id: r.score for r in results}
    assert scores["a::0"] == pytest.approx(0.5)
    assert scores["b::1"] == pytest.approx(0.5)


def test_hybrid_requests_deeper_candidate_pool_than_top_k():
    dense = FakeRetriever([make_chunk(f"d{i}", i) for i in range(30)])
    HybridRetriever(dense, candidates=20).retrieve("q", 3)
    # It must fetch `candidates` deep, not just top_k, so fusion/re-ranking have room.
    assert dense.calls[0][1] == 20


def test_hybrid_applies_reranker_and_truncates():
    dense = FakeRetriever([make_chunk("a"), make_chunk("b", 1), make_chunk("c", 2)])
    reranker = FakeReranker(preferred_order=["c::2", "a::0", "b::1"])
    hybrid = HybridRetriever(dense, reranker=reranker, candidates=20)

    results = hybrid.retrieve("q", 2)
    assert [r.chunk.document_id for r in results] == ["c", "a"]
    query, n_candidates, top_k = reranker.calls[0]
    # The reranker is asked to rank the full candidate pool (not just top_k) so the
    # per-document cap has more than one chunk per document to choose from.
    assert query == "q" and n_candidates == 3 and top_k == 3


class _RaisingReranker:
    """Simulates a reranker whose own retries (D40) are already exhausted -- always
    raises, so HybridRetriever must gracefully fall back to the un-reranked ranking."""

    def rerank(self, query, candidates, top_k):
        raise RuntimeError("cross-encoder inference failed after retries exhausted")


def test_hybrid_falls_back_to_unreranked_when_reranker_exhausts_retries(caplog):
    dense = FakeRetriever([make_chunk("a"), make_chunk("b", 1), make_chunk("c", 2)])
    hybrid = HybridRetriever(dense, reranker=_RaisingReranker(), candidates=20)

    with caplog.at_level("ERROR", logger="kuhaku.tools.rag.retriever"):
        results = hybrid.retrieve("q", 2)

    # Degrades to the un-reranked (fused) ranking, truncated to top_k -- does not
    # propagate the reranker's exception and fail the whole request.
    assert [r.chunk.document_id for r in results] == ["a", "b"]
    assert any("Reranking failed" in r.message for r in caplog.records)


def test_hybrid_skips_reranker_when_no_candidates():
    reranker = FakeReranker()
    hybrid = HybridRetriever(FakeRetriever([]), reranker=reranker, candidates=20)
    assert hybrid.retrieve("q", 3) == []
    assert reranker.calls == []  # never invoked on an empty candidate set


def test_hybrid_top_k_zero():
    dense = FakeRetriever([make_chunk("a")])
    assert HybridRetriever(dense).retrieve("q", 0) == []
    assert dense.calls == []


def test_hybrid_honors_top_k_larger_than_candidate_pool():
    """Asking for more results than `candidates` must still return top_k, not `candidates`."""
    dense = FakeRetriever([make_chunk(f"d{i}", i) for i in range(30)])
    results = HybridRetriever(dense, candidates=5).retrieve("q", 12)
    assert len(results) == 12


def test_hybrid_survives_one_empty_retriever():
    dense = FakeRetriever([make_chunk("a")])
    sparse = FakeRetriever([])  # sparse finds nothing
    results = HybridRetriever(dense, sparse, candidates=20).retrieve("q", 3)
    assert [r.chunk.document_id for r in results] == ["a"]


def test_hybrid_caps_chunks_per_document():
    """At most `max_chunks_per_document` chunks from the same document reach top_k,
    even when that document dominates the ranking (2.3)."""
    dense = FakeRetriever(
        [make_chunk("dominant", i) for i in range(4)] + [make_chunk("other")]
    )
    hybrid = HybridRetriever(dense, candidates=20, max_chunks_per_document=2)
    results = hybrid.retrieve("q", 5)
    assert [r.chunk.document_id for r in results].count("dominant") == 2
    assert "other" in [r.chunk.document_id for r in results]


def test_hybrid_max_chunks_per_document_zero_disables_cap():
    dense = FakeRetriever([make_chunk("dominant", i) for i in range(4)])
    hybrid = HybridRetriever(dense, candidates=20, max_chunks_per_document=0)
    results = hybrid.retrieve("q", 4)
    assert len(results) == 4


def test_hybrid_forwards_doc_type_to_dense_and_sparse():
    dense = FakeRetriever(
        [make_chunk("faq1", doc_type="faq"), make_chunk("runbook1", doc_type="runbook")]
    )
    sparse = FakeRetriever(
        [make_chunk("faq1", doc_type="faq"), make_chunk("runbook1", doc_type="runbook")]
    )
    results = HybridRetriever(dense, sparse, candidates=20).retrieve(
        "q", 3, doc_type="runbook"
    )
    assert [r.chunk.document_id for r in results] == ["runbook1"]
    assert dense.doc_type_calls == ["runbook"]
    assert sparse.doc_type_calls == ["runbook"]


def test_hybrid_forwards_auth_context_to_dense_and_sparse():
    from kuhaku.core.auth import AuthContext

    dense = FakeRetriever([make_chunk("a")])
    sparse = FakeRetriever([make_chunk("a")])
    ctx = AuthContext(identity="u1", is_authenticated=True)
    HybridRetriever(dense, sparse, candidates=20).retrieve("q", 3, auth_context=ctx)
    assert dense.auth_context_calls == [ctx]
    assert sparse.auth_context_calls == [ctx]


# --- strategy labels (used for observability) --------------------------------
def test_dense_retriever_strategy_label():
    assert DenseRetriever(FakeEmbeddings(), FakeVectorStore()).strategy == "dense"


def test_hybrid_strategy_label_dense_only():
    assert HybridRetriever(FakeRetriever([])).strategy == "dense"


def test_hybrid_strategy_label_hybrid():
    hybrid = HybridRetriever(FakeRetriever([]), sparse=FakeRetriever([]))
    assert hybrid.strategy == "hybrid"


def test_hybrid_strategy_label_dense_plus_rerank():
    hybrid = HybridRetriever(FakeRetriever([]), reranker=FakeReranker())
    assert hybrid.strategy == "dense+rerank"


def test_hybrid_strategy_label_hybrid_plus_rerank():
    hybrid = HybridRetriever(
        FakeRetriever([]), sparse=FakeRetriever([]), reranker=FakeReranker()
    )
    assert hybrid.strategy == "hybrid+rerank"


# --- SparseRetriever: orchestration for retrieval="sparse" --------------------------
def test_sparse_retriever_normalizes_raw_scores():
    """Raw, unbounded scores (e.g. BM25's Okapi scores) must land in [0, 1]."""
    sparse = FakeRetriever([make_chunk("hi"), make_chunk("mid", 1), make_chunk("lo", 2)])
    retriever = SparseRetriever(sparse, candidates=20)

    results = retriever.retrieve("q", 3)
    scores = {r.chunk.document_id: r.score for r in results}
    # FakeRetriever scores descend 1.0, 0.9, 0.8 -- min-max normalized over that range.
    assert scores["hi"] == 1.0
    assert scores["lo"] == 0.0
    assert 0.0 < scores["mid"] < 1.0
    # Order is preserved.
    assert [r.chunk.document_id for r in results] == ["hi", "mid", "lo"]


def test_sparse_retriever_single_chunk_no_divide_by_zero():
    sparse = FakeRetriever([make_chunk("only")])
    retriever = SparseRetriever(sparse, candidates=20)

    results = retriever.retrieve("q", 1)
    assert results[0].score == 1.0


def test_sparse_retriever_top_k_zero():
    sparse = FakeRetriever([make_chunk("a")])
    assert SparseRetriever(sparse).retrieve("q", 0) == []


def test_sparse_retriever_forwards_doc_type_and_auth_context():
    from kuhaku.core.auth import AuthContext

    sparse = FakeRetriever([make_chunk("a")])
    ctx = AuthContext(identity="u1", is_authenticated=True)
    SparseRetriever(sparse, candidates=20).retrieve("q", 3, auth_context=ctx, doc_type="faq")
    assert sparse.auth_context_calls == [ctx]
    assert sparse.doc_type_calls == ["faq"]


def test_sparse_retriever_applies_reranker():
    sparse = FakeRetriever([make_chunk("a"), make_chunk("b", 1), make_chunk("c", 2)])
    reranker = FakeReranker(preferred_order=["c::2", "a::0", "b::1"])
    retriever = SparseRetriever(sparse, reranker, candidates=20)

    results = retriever.retrieve("q", 2)
    assert [r.chunk.document_id for r in results] == ["c", "a"]


def test_sparse_retriever_falls_back_to_normalized_score_when_reranker_fails(caplog):
    sparse = FakeRetriever([make_chunk("a"), make_chunk("b", 1)])
    retriever = SparseRetriever(sparse, _RaisingReranker(), candidates=20)

    with caplog.at_level("ERROR", logger="kuhaku.tools.rag.retriever"):
        results = retriever.retrieve("q", 2)

    assert [r.chunk.document_id for r in results] == ["a", "b"]
    assert any("Reranking failed" in r.message for r in caplog.records)


def test_sparse_retriever_strategy_label_no_reranker():
    assert SparseRetriever(FakeRetriever([])).strategy == "sparse"


def test_sparse_retriever_strategy_label_with_reranker():
    assert SparseRetriever(FakeRetriever([]), FakeReranker()).strategy == "sparse+rerank"


def test_sparse_retriever_caps_chunks_per_document():
    sparse = FakeRetriever(
        [make_chunk("dominant", i) for i in range(4)] + [make_chunk("other")]
    )
    retriever = SparseRetriever(sparse, candidates=20, max_chunks_per_document=2)
    results = retriever.retrieve("q", 5)
    assert [r.chunk.document_id for r in results].count("dominant") == 2
    assert "other" in [r.chunk.document_id for r in results]
