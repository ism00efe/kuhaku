"""Tests for the Chroma vector store implementation (real Chroma, temp dir)."""

from __future__ import annotations

import pytest

from kuhaku.tools.rag.vectorstore import ChromaVectorStore
from tests.conftest import make_chunk


def _store(tmp_path, **retry_kwargs) -> ChromaVectorStore:
    return ChromaVectorStore(str(tmp_path / "chroma"), "test_kb", **retry_kwargs)


def test_add_query_count(tmp_path):
    store = _store(tmp_path)
    chunks = [
        make_chunk("errorcodes_payment", 0, text="insufficient funds PAY-1001"),
        make_chunk("faq_general", 0, text="idempotency key usage"),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.add(chunks, embeddings)

    assert store.count() == 2
    results = store.query([1.0, 0.0, 0.0], top_k=2)
    assert results[0].chunk.document_id == "errorcodes_payment"  # nearest to query
    assert results[0].chunk.doc_type == "faq"  # metadata round-trips
    assert results[0].score >= results[1].score  # sorted by similarity


def test_add_empty_is_noop(tmp_path):
    store = _store(tmp_path)
    store.add([], [])
    assert store.count() == 0


def test_reset_clears_collection(tmp_path):
    store = _store(tmp_path)
    store.add([make_chunk("a")], [[1.0, 0.0, 0.0]])
    assert store.count() == 1
    store.reset()
    assert store.count() == 0


def test_persistence_across_instances(tmp_path):
    store = _store(tmp_path)
    store.add([make_chunk("a", text="hello")], [[1.0, 0.0, 0.0]])
    # A new client pointed at the same dir/collection should see the data.
    reopened = ChromaVectorStore(str(tmp_path / "chroma"), "test_kb")
    assert reopened.count() == 1


def test_content_type_round_trips(tmp_path):
    store = _store(tmp_path)
    store.add([make_chunk("a", content_type="table")], [[1.0, 0.0, 0.0]])
    assert store.query([1.0, 0.0, 0.0], top_k=1)[0].chunk.content_type == "table"


def test_content_type_defaults_when_absent_from_stored_metadata(tmp_path):
    """A chunk written before `content_type` existed has no such key in Chroma's
    stored metadata -- query() must default it to "text", not raise a KeyError."""

    store = _store(tmp_path)
    chunk = make_chunk("a")
    metadata = chunk.metadata()
    del metadata["content_type"]
    store._collection.add(  # noqa: SLF001 - simulating a pre-migration row directly
        ids=[chunk.id], embeddings=[[1.0, 0.0, 0.0]],
        documents=[chunk.text], metadatas=[metadata],
    )
    assert store.query([1.0, 0.0, 0.0], top_k=1)[0].chunk.content_type == "text"


# --- FR4: freshness metadata round-trip -----------------------------------------
def test_freshness_fields_round_trip(tmp_path):
    store = _store(tmp_path)
    store.add(
        [make_chunk("a", obsolete=True, effective_date="2025-01-01", expiry_date="2027-01-01")],
        [[1.0, 0.0, 0.0]],
    )
    result = store.query([1.0, 0.0, 0.0], top_k=1)[0].chunk
    assert result.obsolete is True
    assert result.effective_date == "2025-01-01"
    assert result.expiry_date == "2027-01-01"


def test_freshness_fields_default_when_absent_from_stored_metadata(tmp_path):
    """A chunk written before these fields existed has no such keys in Chroma's stored
    metadata -- query() must default to "always fresh", not raise a KeyError."""

    store = _store(tmp_path)
    chunk = make_chunk("a")
    metadata = chunk.metadata()
    del metadata["effective_date"], metadata["obsolete"], metadata["expiry_date"]
    store._collection.add(  # noqa: SLF001 - simulating a pre-migration row directly
        ids=[chunk.id], embeddings=[[1.0, 0.0, 0.0]],
        documents=[chunk.text], metadatas=[metadata],
    )
    result = store.query([1.0, 0.0, 0.0], top_k=1)[0].chunk
    assert result.obsolete is False
    assert result.effective_date == "" and result.expiry_date == ""
    assert result.is_fresh() is True


# --- Retry (D40): query() only, not add/count/reset -----------------------------
def test_query_retries_transient_failure_then_succeeds(tmp_path, monkeypatch):
    store = _store(tmp_path, retry_backoff_seconds=0.001)
    store.add([make_chunk("a", text="hello")], [[1.0, 0.0, 0.0]])

    original_query = store._collection.query  # noqa: SLF001
    calls: list[int] = []

    def flaky_query(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient lock contention")
        return original_query(*args, **kwargs)

    monkeypatch.setattr(store._collection, "query", flaky_query)  # noqa: SLF001
    results = store.query([1.0, 0.0, 0.0], top_k=1)

    assert len(calls) == 2
    assert results[0].chunk.document_id == "a"


def test_query_exhausts_retries_and_raises_original_exception(tmp_path, monkeypatch):
    store = _store(tmp_path, retry_max_attempts=2, retry_backoff_seconds=0.001)
    store.add([make_chunk("a")], [[1.0, 0.0, 0.0]])

    calls: list[int] = []

    def always_fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("persistent lock contention")

    monkeypatch.setattr(store._collection, "query", always_fail)  # noqa: SLF001
    with pytest.raises(RuntimeError, match="persistent lock contention"):
        store.query([1.0, 0.0, 0.0], top_k=1)
    assert len(calls) == 2


def test_query_retry_disabled_bypasses_retry(tmp_path, monkeypatch):
    store = _store(tmp_path, retry_enabled=False)
    store.add([make_chunk("a")], [[1.0, 0.0, 0.0]])

    calls: list[int] = []

    def always_fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("persistent lock contention")

    monkeypatch.setattr(store._collection, "query", always_fail)  # noqa: SLF001
    with pytest.raises(RuntimeError):
        store.query([1.0, 0.0, 0.0], top_k=1)
    assert len(calls) == 1


# --- get_by_document_id (D47) ----------------------------------------------------
def test_get_by_document_id_returns_all_chunks_for_a_document(tmp_path):
    store = _store(tmp_path)
    store.add(
        [
            make_chunk("doc_a", 0, text="first chunk"),
            make_chunk("doc_a", 1, text="second chunk"),
            make_chunk("doc_b", 0, text="other document"),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    chunks = store.get_by_document_id("doc_a")
    assert {c.id for c in chunks} == {"doc_a::0", "doc_a::1"}
    assert all(c.document_id == "doc_a" for c in chunks)


def test_get_by_document_id_returns_empty_list_for_unknown_document(tmp_path):
    store = _store(tmp_path)
    store.add([make_chunk("doc_a")], [[1.0, 0.0, 0.0]])
    assert store.get_by_document_id("no_such_doc") == []


# --- get_by_ids (D49) -------------------------------------------------------------
def test_get_by_ids_returns_the_right_chunks(tmp_path):
    store = _store(tmp_path)
    store.add(
        [
            make_chunk("doc_a", 0, text="first chunk"),
            make_chunk("doc_a", 1, text="second chunk"),
            make_chunk("doc_b", 0, text="other document"),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    chunks = store.get_by_ids(["doc_a::0", "doc_b::0"])
    assert {c.id for c in chunks} == {"doc_a::0", "doc_b::0"}


def test_get_by_ids_omits_missing_ids_silently(tmp_path):
    store = _store(tmp_path)
    store.add([make_chunk("doc_a", 0, text="only chunk")], [[1.0, 0.0, 0.0]])
    chunks = store.get_by_ids(["doc_a::0", "no_such_chunk::0"])
    assert {c.id for c in chunks} == {"doc_a::0"}


def test_get_by_ids_empty_input_returns_empty_list_without_querying(tmp_path):
    store = _store(tmp_path)
    store.add([make_chunk("doc_a")], [[1.0, 0.0, 0.0]])
    assert store.get_by_ids([]) == []


# --- iter_chunks (Feature 1: enumerate every chunk the store holds) --------------
def test_iter_chunks_round_trips_id_text_and_metadata(tmp_path):
    store = _store(tmp_path)
    chunks = [
        make_chunk("doc_a", 0, text="first chunk", obsolete=True, effective_date="2025-01-01"),
        make_chunk("doc_b", 0, text="second chunk", content_type="table"),
    ]
    store.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    read_back = {c.id: c for c in store.iter_chunks()}
    assert set(read_back) == {c.id for c in chunks}
    for original in chunks:
        got = read_back[original.id]
        assert got.text == original.text
        assert got.metadata() == original.metadata()


def test_iter_chunks_empty_store_yields_nothing(tmp_path):
    store = _store(tmp_path)
    assert list(store.iter_chunks()) == []


def test_iter_chunks_pages_through_collections_larger_than_page_size(tmp_path):
    store = _store(tmp_path)
    chunks = [make_chunk(f"doc_{i}", text=f"content {i}") for i in range(12)]
    embeddings = [[float(i), 0.0, 0.0] for i in range(12)]
    store.add(chunks, embeddings)

    got = list(store.iter_chunks(page_size=5))
    assert {c.id for c in got} == {c.id for c in chunks}
    assert len(got) == 12


def test_iter_chunks_refreshes_stale_collection(tmp_path):
    store = _store(tmp_path)
    store.add([make_chunk("a")], [[1.0, 0.0, 0.0]])
    store._client.delete_collection(store._collection_name)  # noqa: SLF001
    assert list(store.iter_chunks()) == []


def test_iter_chunks_reflects_add_semantics_on_id_collision(tmp_path):
    """``add()`` ignores a duplicate id rather than overwriting it (verified against the
    installed Chroma client) -- ``iter_chunks()`` must faithfully reflect that
    store-level behavior, not paper over it, since it's the only enumeration path a
    consumer like BM25 index construction has."""

    store = _store(tmp_path)
    store.add([make_chunk("a", text="original")], [[1.0, 0.0, 0.0]])
    store.add([make_chunk("a", text="colliding")], [[0.0, 1.0, 0.0]])  # same id "a::0"

    chunks = list(store.iter_chunks())
    assert len(chunks) == 1
    assert chunks[0].text == "original"


def test_stale_collection_refreshes_automatically(tmp_path):
    """If the underlying Chroma collection is deleted or reset externally, count/query/add
    refreshes the stale collection reference automatically without failing."""

    store = _store(tmp_path)
    store.add([make_chunk("a")], [[1.0, 0.0, 0.0]])
    assert store.count() == 1

    # Simulate external deletion of collection from Chroma client
    store._client.delete_collection(store._collection_name)

    # count() and query() should auto-recover and return valid collection
    assert store.count() == 0
    store.add([make_chunk("b")], [[0.0, 1.0, 0.0]])
    assert store.count() == 1

