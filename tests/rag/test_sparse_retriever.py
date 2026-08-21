"""Tests for the hand-rolled BM25 sparse retriever."""

from __future__ import annotations

from kuhaku.tools.rag.sparse_retriever import (
    BM25Retriever,
    build_bm25_from_store,
    tokenize,
)
from tests.conftest import FakeVectorStore, make_chunk


# --- tokenizer --------------------------------------------------------------
def test_tokenize_lowercases_and_splits():
    assert tokenize("Gateway TIMEOUT occurred") == ["gateway", "timeout", "occurred"]


def test_tokenize_splits_error_codes():
    # The numeric part is what makes an error code discriminative for BM25.
    assert tokenize("PAY-6006") == ["pay", "6006"]


def test_tokenize_preserves_turkish_characters():
    assert tokenize("ödeme başarısız") == ["ödeme", "başarısız"]


def test_tokenize_empty():
    assert tokenize("") == []


# --- retrieval --------------------------------------------------------------
def _corpus() -> list[BM25Retriever]:
    return [
        make_chunk("doc_1001", text="Error PAY-1001 insufficient funds issuer decline"),
        make_chunk("doc_6006", text="Error PAY-6006 gateway timeout retry with idempotency"),
        make_chunk("doc_webhook", text="Webhook signature verification uses HMAC SHA256"),
    ]


def test_exact_error_code_ranks_first():
    retriever = BM25Retriever(_corpus())
    results = retriever.retrieve("PAY-6006", top_k=3)
    assert results[0].chunk.document_id == "doc_6006"


def test_lexical_match_on_distinct_term():
    retriever = BM25Retriever(_corpus())
    results = retriever.retrieve("HMAC signature", top_k=3)
    assert results[0].chunk.document_id == "doc_webhook"


def test_only_matching_documents_returned():
    retriever = BM25Retriever(_corpus())
    results = retriever.retrieve("webhook", top_k=10)
    # Non-matching docs score 0 and are excluded entirely.
    assert [r.chunk.document_id for r in results] == ["doc_webhook"]


def test_no_match_returns_empty():
    retriever = BM25Retriever(_corpus())
    assert retriever.retrieve("kubernetes helm chart", top_k=5) == []


def test_doc_type_filter_excludes_other_types():
    chunks = [
        make_chunk("doc_faq", text="webhook signature guide", doc_type="faq"),
        make_chunk("doc_runbook", text="webhook signature guide", doc_type="runbook"),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("webhook signature", top_k=10, doc_type="runbook")
    assert [r.chunk.document_id for r in results] == ["doc_runbook"]


def test_doc_type_none_is_unfiltered():
    chunks = [
        make_chunk("doc_faq", text="webhook signature guide", doc_type="faq"),
        make_chunk("doc_runbook", text="webhook signature guide", doc_type="runbook"),
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("webhook signature", top_k=10, doc_type=None)
    assert {r.chunk.document_id for r in results} == {"doc_faq", "doc_runbook"}


def test_top_k_limits_results():
    retriever = BM25Retriever(_corpus())
    assert len(retriever.retrieve("error pay", top_k=1)) == 1


def test_top_k_zero_or_negative():
    retriever = BM25Retriever(_corpus())
    assert retriever.retrieve("error", top_k=0) == []


def test_empty_corpus_is_safe():
    retriever = BM25Retriever([])
    assert retriever.count() == 0
    assert retriever.retrieve("anything", top_k=5) == []


def test_bm25_retriever_strategy_label():
    assert BM25Retriever([]).strategy == "sparse"


def test_scores_are_positive_and_descending():
    retriever = BM25Retriever(_corpus())
    results = retriever.retrieve("error pay timeout", top_k=3)
    scores = [r.score for r in results]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


# --- FR4: freshness filtering ---------------------------------------------------
def test_bm25_filters_out_obsolete_chunks():
    chunks = [
        make_chunk("fresh", text="ödeme hatası PAY-1001"),
        make_chunk("stale", 1, text="ödeme hatası PAY-1001", obsolete=True),
    ]
    results = BM25Retriever(chunks).retrieve("PAY-1001", top_k=5)
    assert [r.chunk.document_id for r in results] == ["fresh"]


def test_bm25_filters_out_expired_chunks():
    chunks = [
        make_chunk("fresh", text="ödeme hatası PAY-1001"),
        make_chunk("expired", 1, text="ödeme hatası PAY-1001", expiry_date="2020-01-01"),
    ]
    results = BM25Retriever(chunks).retrieve("PAY-1001", top_k=5)
    assert [r.chunk.document_id for r in results] == ["fresh"]


# --- auth_context pass-through: BM25Retriever filters nothing based on it ----------
def test_bm25_performs_no_filtering_based_on_auth_context():
    """Same contract as DenseRetriever: BM25Retriever does not interpret auth_context --
    authorization decisions live in RAGEngine's optional AuthorizationPolicy
    (kuhaku.core.auth), not in the retriever layer."""

    from kuhaku.core.auth import AuthContext

    chunks = [make_chunk("a", text="ödeme hatası PAY-1001")]
    unauth = BM25Retriever(chunks).retrieve("PAY-1001", top_k=5)
    authed = BM25Retriever(chunks).retrieve(
        "PAY-1001", top_k=5, auth_context=AuthContext(identity="u1", is_authenticated=True)
    )
    assert {r.chunk.document_id for r in unauth} == {r.chunk.document_id for r in authed}


def test_rarer_term_outranks_common_term():
    """IDF should favour the document matching the rare term."""
    chunks = [
        make_chunk("common_a", text="payment payment payment"),
        make_chunk("common_b", text="payment payment payment"),
        make_chunk("rare", text="payment chargeback"),
    ]
    results = BM25Retriever(chunks).retrieve("payment chargeback", top_k=3)
    assert results[0].chunk.document_id == "rare"


# --- store-backed factory ----------------------------------------------------
# Sanitization coverage lives in tests/security/test_pii_leak_scan.py, which drives the
# real ingest() -> store -> build_bm25_from_store path end to end. The
# dense/sparse chunker-mismatch hazard the old corpus-reading factory carried is now
# structurally impossible: both sides read the exact same Chunk objects out of the same
# store, so there is only ever one chunking of a given document (see
# StoreBackedBM25Retriever's docstring).
def test_build_bm25_from_store_indexes_the_stores_current_chunks():
    store = FakeVectorStore([make_chunk("errorcodes_payment", text="insufficient funds PAY-1001")])
    retriever = build_bm25_from_store(store)
    assert retriever.count() == 1
    assert retriever.retrieve("PAY-1001", top_k=5)


def test_build_bm25_from_store_empty_store_is_safe():
    retriever = build_bm25_from_store(FakeVectorStore())
    assert retriever.count() == 0
    assert retriever.retrieve("anything", top_k=5) == []


def test_build_bm25_from_store_tolerates_duplicate_chunk_ids():
    """Not every VectorStore need dedupe on id the way Chroma's own add() does (see
    test_vectorstore.py's collision test) -- BM25 construction itself must not crash if
    a store's enumeration ever yields two chunks sharing an id; it simply indexes both
    by list position, same as any other pair of chunks."""

    store = FakeVectorStore(
        [make_chunk("a", text="alpha one"), make_chunk("a", text="alpha two")]
    )
    retriever = build_bm25_from_store(store)
    assert retriever.count() == 2
    assert retriever.retrieve("alpha", top_k=5)


def test_refresh_picks_up_chunks_ingested_after_construction():
    store = FakeVectorStore([make_chunk("a", text="alpha")])
    retriever = build_bm25_from_store(store)
    assert retriever.retrieve("beta", top_k=5) == []

    store.add([make_chunk("b", text="beta")], [[0.0, 0.0, 0.0]])
    # Without refresh(), the cached index must not silently see the new chunk.
    assert retriever.retrieve("beta", top_k=5) == []

    retriever.refresh()
    results = retriever.retrieve("beta", top_k=5)
    assert [r.chunk.document_id for r in results] == ["b"]


def test_repeated_ingests_keep_the_index_correct():
    store = FakeVectorStore([make_chunk("seed", text="seed term")])
    retriever = build_bm25_from_store(store)
    retriever.retrieve("seed", top_k=5)  # force the first build

    for i in range(5):
        store.add([make_chunk(f"doc_{i}", text=f"term{i}")], [[0.0, 0.0, 0.0]])
        retriever.refresh()

    assert retriever.count() == 6
    for i in range(5):
        results = retriever.retrieve(f"term{i}", top_k=5)
        assert [r.chunk.document_id for r in results] == [f"doc_{i}"]


def test_refresh_defers_rebuild_until_the_next_query(monkeypatch):
    """Multiple refresh() calls between two queries must collapse into a single
    rebuild, not one rebuild per refresh() call -- otherwise a run of N ingests with
    no interleaved queries costs O(N * corpus_size), quadratic in N (see
    StoreBackedBM25Retriever.refresh)."""

    import kuhaku.tools.rag.sparse_retriever as sparse_retriever_module

    build_calls: list[int] = []
    original_init = sparse_retriever_module.BM25Retriever.__init__

    def counting_init(self, chunks, **kwargs):
        build_calls.append(len(chunks))
        original_init(self, chunks, **kwargs)

    monkeypatch.setattr(sparse_retriever_module.BM25Retriever, "__init__", counting_init)

    store = FakeVectorStore([make_chunk("a", text="alpha")])
    retriever = build_bm25_from_store(store)
    retriever.retrieve("alpha", top_k=5)
    assert len(build_calls) == 1

    for i in range(5):
        store.add([make_chunk(f"doc_{i}", text=f"term{i}")], [[0.0, 0.0, 0.0]])
        retriever.refresh()
    assert len(build_calls) == 1  # refresh() alone never rebuilds

    retriever.retrieve("term3", top_k=5)
    assert len(build_calls) == 2  # one rebuild covers all 5 pending refreshes
