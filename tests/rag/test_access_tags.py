"""Tests for document-level access filtering (access_tags).

Covers the security-critical guarantees end to end: tag storage/round-trip (Feature 1),
per-strategy entitlement enforcement and the rank-cliff regression (Feature 3), and the
default tag vocabulary (Feature 6). Cache isolation and the unentitled/no-match
indistinguishability property live in test_engine.py; the public-facade acceptance test
lives in test_rag_facade.py.
"""

from __future__ import annotations

import math

from kuhaku.core.auth import AuthContext
from kuhaku.tools.rag.models import (
    ACCESS_TAG_INTERNAL,
    ACCESS_TAG_PUBLIC,
    ACCESS_TAG_RESTRICTED,
    Chunk,
)
from kuhaku.tools.rag.retriever import DenseRetriever, HybridRetriever, is_entitled
from kuhaku.tools.rag.sparse_retriever import BM25Retriever
from kuhaku.tools.rag.vectorstore import ChromaVectorStore
from tests.conftest import FakeEmbeddings, FakeRetriever, FakeVectorStore, make_chunk


# --- Feature 1: storage round-trip (real Chroma) -------------------------------------
def test_tagged_chunk_round_trips_through_chroma(tmp_path):
    store = ChromaVectorStore(str(tmp_path / "chroma"), "tags")
    chunk = make_chunk("hr_doc", access_tags=("people_ops", "legal"))
    store.add([chunk], [[1.0, 0.0, 0.0]])

    read_back = {c.id: c for c in store.iter_chunks()}[chunk.id]
    assert set(read_back.access_tags) == {"people_ops", "legal"}
    assert read_back.metadata() == chunk.metadata()


def test_untagged_chunk_round_trips_as_untagged(tmp_path):
    store = ChromaVectorStore(str(tmp_path / "chroma"), "tags")
    chunk = make_chunk("public_doc")
    store.add([chunk], [[1.0, 0.0, 0.0]])

    read_back = {c.id: c for c in store.iter_chunks()}[chunk.id]
    assert read_back.access_tags == ()


def test_none_and_empty_access_tags_store_identically(tmp_path):
    """`Chunk(access_tags=())` (the dataclass default, standing in for `None` at the
    ingestion boundary -- see ingestion._normalize_access_tags) must be indistinguishable
    on disk from any other untagged chunk."""

    store = ChromaVectorStore(str(tmp_path / "chroma"), "tags")
    a = make_chunk("a", access_tags=())
    b = make_chunk("b", access_tags=())
    store.add([a, b], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    rows = {c.id: c for c in store.iter_chunks()}
    assert rows["a::0"].access_tags == () and rows["b::0"].access_tags == ()
    assert rows["a::0"].metadata()["access_tags"] == rows["b::0"].metadata()["access_tags"]


def test_untagged_chunk_stores_nothing_that_could_accidentally_match_a_filter(tmp_path):
    """An untagged chunk must never satisfy a `$contains` clause for any tag string --
    including the empty string or a tag nobody would ever legitimately use."""

    store = ChromaVectorStore(str(tmp_path / "chroma"), "tags")
    store.add([make_chunk("untagged")], [[1.0, 0.0, 0.0]])

    for candidate_tag in ("", "people_ops", "public", "*"):
        matches = store.query(
            [1.0, 0.0, 0.0], top_k=5, where={"access_tags": {"$contains": candidate_tag}}
        )
        assert matches == []


# --- Feature 3: per-strategy enforcement ----------------------------------------------
def _hr_and_public_store() -> FakeVectorStore:
    return FakeVectorStore(
        [
            make_chunk("hr_doc", access_tags=("people_ops",), text="salary bands are A-E"),
            make_chunk("multi_tag_doc", access_tags=("people_ops", "legal"), text="policy text"),
            make_chunk("public_doc", text="deploy runbook steps"),
        ]
    )


def test_dense_tagged_chunk_hidden_from_non_matching_role():
    store = _hr_and_public_store()
    retriever = DenseRetriever(FakeEmbeddings(), store)
    results = retriever.retrieve(
        "q", 10, auth_context=AuthContext(identity="u", roles=("engineering",))
    )
    ids = {r.chunk.document_id for r in results}
    assert "hr_doc" not in ids and "multi_tag_doc" not in ids


def test_dense_untagged_chunk_visible_to_any_role():
    store = _hr_and_public_store()
    retriever = DenseRetriever(FakeEmbeddings(), store)
    for ctx in (None, AuthContext(identity="u", roles=()), AuthContext(identity="u", roles=("anything",))):
        results = retriever.retrieve("q", 10, auth_context=ctx)
        assert "public_doc" in {r.chunk.document_id for r in results}


def test_dense_partial_tag_match_grants_access():
    """A chunk tagged with several tags is visible when *any one* of them matches --
    flat set intersection, not "all tags must match"."""

    store = _hr_and_public_store()
    retriever = DenseRetriever(FakeEmbeddings(), store)
    results = retriever.retrieve(
        "q", 10, auth_context=AuthContext(identity="u", roles=("legal",))
    )
    assert "multi_tag_doc" in {r.chunk.document_id for r in results}
    assert "hr_doc" not in {r.chunk.document_id for r in results}  # legal != people_ops


def test_dense_empty_roles_sees_only_untagged():
    store = _hr_and_public_store()
    retriever = DenseRetriever(FakeEmbeddings(), store)
    results = retriever.retrieve(
        "q", 10, auth_context=AuthContext(identity="u", roles=())
    )
    assert {r.chunk.document_id for r in results} == {"public_doc"}


def _hr_and_public_chunks() -> list[Chunk]:
    return [
        make_chunk("hr_doc", access_tags=("people_ops",), text="salary salary salary bands"),
        make_chunk("multi_tag_doc", access_tags=("people_ops", "legal"), text="salary policy"),
        make_chunk("public_doc", text="salary mentioned once in a runbook"),
    ]


def test_sparse_tagged_chunk_hidden_from_non_matching_role():
    retriever = BM25Retriever(_hr_and_public_chunks())
    results = retriever.retrieve(
        "salary", top_k=10, auth_context=AuthContext(identity="u", roles=("engineering",))
    )
    ids = {r.chunk.document_id for r in results}
    assert "hr_doc" not in ids and "multi_tag_doc" not in ids


def test_sparse_untagged_chunk_visible_to_any_role():
    retriever = BM25Retriever(_hr_and_public_chunks())
    for ctx in (None, AuthContext(identity="u", roles=()), AuthContext(identity="u", roles=("anything",))):
        results = retriever.retrieve("salary", top_k=10, auth_context=ctx)
        assert "public_doc" in {r.chunk.document_id for r in results}


def test_sparse_partial_tag_match_grants_access():
    retriever = BM25Retriever(_hr_and_public_chunks())
    results = retriever.retrieve(
        "salary", top_k=10, auth_context=AuthContext(identity="u", roles=("legal",))
    )
    assert "multi_tag_doc" in {r.chunk.document_id for r in results}
    assert "hr_doc" not in {r.chunk.document_id for r in results}


def test_sparse_empty_roles_sees_only_untagged():
    retriever = BM25Retriever(_hr_and_public_chunks())
    results = retriever.retrieve(
        "salary", top_k=10, auth_context=AuthContext(identity="u", roles=())
    )
    assert {r.chunk.document_id for r in results} == {"public_doc"}


def test_hybrid_tagged_chunk_hidden_from_non_matching_role():
    dense = FakeRetriever(_hr_and_public_chunks())
    sparse = FakeRetriever(_hr_and_public_chunks())
    hybrid = HybridRetriever(dense, sparse, candidates=20)
    results = hybrid.retrieve(
        "q", 10, auth_context=AuthContext(identity="u", roles=("engineering",))
    )
    ids = {r.chunk.document_id for r in results}
    assert "hr_doc" not in ids and "multi_tag_doc" not in ids


def test_hybrid_untagged_chunk_visible_to_any_role():
    dense = FakeRetriever(_hr_and_public_chunks())
    sparse = FakeRetriever(_hr_and_public_chunks())
    hybrid = HybridRetriever(dense, sparse, candidates=20)
    for ctx in (None, AuthContext(identity="u", roles=())):
        results = hybrid.retrieve("q", 10, auth_context=ctx)
        assert "public_doc" in {r.chunk.document_id for r in results}


def test_hybrid_partial_tag_match_grants_access():
    dense = FakeRetriever(_hr_and_public_chunks())
    sparse = FakeRetriever(_hr_and_public_chunks())
    hybrid = HybridRetriever(dense, sparse, candidates=20)
    results = hybrid.retrieve(
        "q", 10, auth_context=AuthContext(identity="u", roles=("legal",))
    )
    assert "multi_tag_doc" in {r.chunk.document_id for r in results}
    assert "hr_doc" not in {r.chunk.document_id for r in results}


def test_hybrid_empty_roles_sees_only_untagged():
    dense = FakeRetriever(_hr_and_public_chunks())
    sparse = FakeRetriever(_hr_and_public_chunks())
    hybrid = HybridRetriever(dense, sparse, candidates=20)
    results = hybrid.retrieve(
        "q", 10, auth_context=AuthContext(identity="u", roles=())
    )
    assert {r.chunk.document_id for r in results} == {"public_doc"}


def test_hybrid_never_lets_a_restricted_chunk_enter_fusion():
    """Both sides rank the restricted chunk first; it must still never appear in the
    fused result -- proving it was filtered independently on each side before RRF,
    never merely dropped after fusion."""

    restricted = make_chunk("restricted", access_tags=("hr",))
    eligible = make_chunk("eligible")
    dense = FakeRetriever([restricted, eligible])
    sparse = FakeRetriever([restricted, eligible])
    hybrid = HybridRetriever(dense, sparse, candidates=20)

    results = hybrid.retrieve(
        "q", 2, auth_context=AuthContext(identity="u", roles=("engineering",))
    )
    assert [r.chunk.document_id for r in results] == ["eligible"]


# --- Feature 3: rank-cliff regression (the critical Definition of Done) --------------
def _unit_vector(angle_degrees: float) -> list[float]:
    theta = math.radians(angle_degrees)
    return [math.cos(theta), math.sin(theta)]


class _FixedQueryEmbedder:
    """Returns the same query vector regardless of input text -- lets the test control
    exactly which stored chunks rank nearest, independent of any real embedding model."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError("chunks in this test are added with explicit embeddings")

    def embed_query(self, text: str) -> list[float]:
        return self._vector


def test_dense_rank_cliff_regression(tmp_path):
    """The top 10 nearest-neighbor chunks are all restricted; 5 more-distant eligible
    chunks exist beyond them. An entitled-to-something user asking for top_k=4 must
    still get a full 4 results drawn from the eligible set -- proving entitlement
    narrows the candidate set the ANN search itself considers, not the final top-k list
    after the fact (which would return fewer than top_k, or nothing)."""

    store = ChromaVectorStore(str(tmp_path / "chroma"), "rank_cliff")
    restricted_chunks = [
        make_chunk(f"restricted_{i}", access_tags=("hr",), text=f"restricted {i}")
        for i in range(10)
    ]
    eligible_chunks = [make_chunk(f"eligible_{i}", text=f"eligible {i}") for i in range(5)]
    # Restricted chunks sit at small angles from the query direction (very similar);
    # eligible chunks sit further away -- an unfiltered top-4 search returns only
    # restricted chunks (asserted below as a sanity check on the test setup itself).
    store.add(restricted_chunks, [_unit_vector(i + 1) for i in range(10)])
    store.add(eligible_chunks, [_unit_vector(30 + i) for i in range(5)])

    unfiltered = store.query([1.0, 0.0], top_k=4)
    assert all(c.chunk.document_id.startswith("restricted") for c in unfiltered)

    retriever = DenseRetriever(_FixedQueryEmbedder([1.0, 0.0]), store)
    results = retriever.retrieve(
        "query", 4, auth_context=AuthContext(identity="u", roles=("engineering",))
    )

    assert len(results) == 4
    assert all(r.chunk.document_id.startswith("eligible") for r in results)


def test_sparse_rank_cliff_regression():
    """Same regression as dense, for BM25: 10 restricted chunks score higher (repeat the
    query term) than 5 eligible chunks that still match, but an entitled user requesting
    top_k=4 must still get 4 eligible results, not fewer."""

    restricted = [
        make_chunk(f"restricted_{i}", access_tags=("hr",), text="salary salary salary salary bands")
        for i in range(10)
    ]
    eligible = [make_chunk(f"eligible_{i}", text="salary mentioned once") for i in range(5)]
    retriever = BM25Retriever(restricted + eligible)

    # Sanity check on the test setup itself: with full visibility (a caller entitled to
    # both the restricted and the untagged content -- everything here is either
    # untagged or tagged "hr"), raw BM25 ranking surfaces only the higher-scoring
    # restricted chunks.
    fully_entitled = retriever.retrieve(
        "salary", top_k=4, auth_context=AuthContext(identity="u", roles=("hr",))
    )
    assert all(c.chunk.document_id.startswith("restricted") for c in fully_entitled)

    results = retriever.retrieve(
        "salary", top_k=4, auth_context=AuthContext(identity="u", roles=("engineering",))
    )
    assert len(results) == 4
    assert all(r.chunk.document_id.startswith("eligible") for r in results)


def test_hybrid_rank_cliff_regression(tmp_path):
    """Same regression as dense/sparse (this is the new default retrieval strategy, so it
    needs the same coverage), through genuine dense+sparse fusion: restricted chunks rank
    first on *both* sides (nearest by embedding, highest by BM25 term repetition), so
    fusion would surface only them too, unless entitlement narrows each side's candidate
    set before ranking/fusion ever runs -- not merely after."""

    store = ChromaVectorStore(str(tmp_path / "chroma"), "hybrid_rank_cliff")
    restricted_chunks = [
        make_chunk(
            f"restricted_{i}", access_tags=("hr",), text="salary salary salary salary bands"
        )
        for i in range(10)
    ]
    eligible_chunks = [
        make_chunk(f"eligible_{i}", text="salary mentioned once") for i in range(5)
    ]
    store.add(restricted_chunks, [_unit_vector(i + 1) for i in range(10)])
    store.add(eligible_chunks, [_unit_vector(30 + i) for i in range(5)])

    dense = DenseRetriever(_FixedQueryEmbedder([1.0, 0.0]), store)
    sparse = BM25Retriever(restricted_chunks + eligible_chunks)
    hybrid = HybridRetriever(dense, sparse, candidates=20)

    # Sanity check on the test setup itself: with full visibility, both the ANN search
    # and BM25 rank every restricted chunk ahead of every eligible one, so fusion does
    # too.
    fully_entitled = hybrid.retrieve(
        "salary", 4, auth_context=AuthContext(identity="u", roles=("hr",))
    )
    assert all(r.chunk.document_id.startswith("restricted") for r in fully_entitled)

    results = hybrid.retrieve(
        "salary", 4, auth_context=AuthContext(identity="u", roles=("engineering",))
    )
    assert len(results) == 4
    assert all(r.chunk.document_id.startswith("eligible") for r in results)


# --- is_entitled: the single predicate -----------------------------------------------
def test_is_entitled_untagged_always_true():
    assert is_entitled(make_chunk("a"), None) is True
    assert is_entitled(make_chunk("a"), AuthContext(identity="u", roles=())) is True
    assert is_entitled(make_chunk("a"), AuthContext(identity="u", roles=("x",))) is True


def test_is_entitled_tagged_requires_role_intersection():
    chunk = make_chunk("a", access_tags=("hr", "legal"))
    assert is_entitled(chunk, None) is False
    assert is_entitled(chunk, AuthContext(identity="u", roles=())) is False
    assert is_entitled(chunk, AuthContext(identity="u", roles=("engineering",))) is False
    assert is_entitled(chunk, AuthContext(identity="u", roles=("legal",))) is True
    assert is_entitled(chunk, AuthContext(identity="u", roles=("legal", "hr"))) is True


# --- Feature 6: default tag vocabulary, plain strings only ---------------------------
def test_default_vocabulary_constants_are_plain_strings():
    assert {ACCESS_TAG_PUBLIC, ACCESS_TAG_INTERNAL, ACCESS_TAG_RESTRICTED} == {
        "public", "internal", "restricted",
    }


def test_arbitrary_string_works_identically_to_a_default_constant():
    """A caller-invented tag must behave exactly like one of the shipped constants --
    there is no validation, enum, or registry restricting what a tag string can be."""

    via_constant = make_chunk("a", access_tags=(ACCESS_TAG_RESTRICTED,))
    via_custom_string = make_chunk("b", access_tags=("insan_kaynaklari",))
    ctx = AuthContext(identity="u", roles=("insan_kaynaklari",))

    assert is_entitled(via_constant, ctx) is False  # role doesn't match "restricted"
    assert is_entitled(via_custom_string, ctx) is True  # role matches the custom tag

    matching_ctx = AuthContext(identity="u", roles=(ACCESS_TAG_RESTRICTED,))
    assert is_entitled(via_constant, matching_ctx) is True
