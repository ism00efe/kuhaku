"""Retriever abstraction, dense adapter, and hybrid fusion.

Introduces a single ``Retriever`` seam so the engine can be given dense-only, hybrid, or
hybrid+re-ranked retrieval without changing the existing ``VectorStore`` /
``EmbeddingProvider`` abstractions — ``DenseRetriever`` is a thin *adapter* over that
existing pair.

Fusion uses Reciprocal Rank Fusion (RRF):

    score(d) = Σ_r  1 / (k + rank_r(d))

RRF combines rankings, not scores, so it needs no score normalization between a cosine
similarity and a BM25 score — which is exactly why it is the standard choice here.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from kuhaku.core.auth import AuthContext
from kuhaku.core.observability import instrumented_step
from kuhaku.tools.rag.models import ACCESS_TAGS_NONE_KEY, Chunk, RetrievedChunk

from .embeddings import EmbeddingProvider, EmbeddingServiceError
from .metrics import ACTIVE_DOCUMENTS
from .reranker import Reranker
from .vectorstore import VectorStore, VectorStoreError

logger = logging.getLogger("kuhaku.tools.rag.retriever")


@runtime_checkable
class Retriever(Protocol):
    """Returns the most relevant chunks for a (already sanitized) query.

    ``auth_context`` gates document-level access: every built-in retriever below enforces
    ``is_entitled`` (see below) before ranking/truncating to ``top_k`` -- a chunk the
    caller is not entitled to see never enters the ranked result at all, let alone the
    LLM prompt. ``auth_context=None`` is not "no restriction"; it means "no roles", so
    only untagged chunks are visible (see ``is_entitled``'s own docstring). Filtering is
    unconditional: there is no argument, anywhere, that turns it off.
    """

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        auth_context: AuthContext | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]: ...


def refresh_retriever(retriever: object) -> None:
    """Tell ``retriever`` that new chunks may have been written to its backing store, if
    it tracks any cached state that would otherwise go stale.

    Duck-typed rather than a required ``Retriever`` protocol method: ``DenseRetriever``
    always queries the live vector store and has nothing to refresh, and callers (see
    ``RAGEngine.ingest_document``) must stay decoupled from which concrete retriever
    they hold -- adding a mandatory method here would force every ``Retriever``,
    dense included, to implement a no-op. A no-``refresh`` retriever is simply left
    alone.
    """

    refresh = getattr(retriever, "refresh", None)
    if callable(refresh):
        refresh()


def _filter_by_doc_type(
    items: list[RetrievedChunk], doc_type: str | None
) -> list[RetrievedChunk]:
    """Keep only chunks whose parent document is ``doc_type`` (4.1).

    ``doc_type=None`` (the default everywhere) is a no-op. Both retrieval paths filter
    before truncating to ``top_k`` (dense through the store's ``where``, BM25 before
    ranking), so off-type chunks never crowd matching ones out.
    """

    if doc_type is None:
        return items
    return [item for item in items if item.chunk.doc_type == doc_type]


def is_entitled(chunk: Chunk, auth_context: AuthContext | None) -> bool:
    """Document-level access filtering: the single rule every retrieval strategy
    enforces, so it needs defining in exactly one place (dense translates the same rule
    into a Chroma ``where`` filter -- see ``_entitlement_where`` below -- rather than
    calling this directly, but the two must always agree).

    Flat set intersection only (no hierarchy, no ordering, no tag implies another): a
    chunk with no ``access_tags`` is visible to everyone -- tagging is opt-in. A tagged
    chunk is visible only when at least one of its tags also appears in
    ``auth_context.roles``. ``auth_context=None`` satisfies no tag at all (equivalent to
    an ``AuthContext`` with empty ``roles``), so it can only ever see untagged chunks --
    there is no separate "disabled" state where filtering is skipped.

    This is the seam a future caller-supplied predicate would replace: every call site
    below calls this one function rather than inlining the set-intersection check.
    """

    if not chunk.access_tags:
        return True
    if auth_context is None:
        return False
    return not set(chunk.access_tags).isdisjoint(auth_context.roles)


def _entitlement_where(auth_context: AuthContext | None) -> dict[str, Any]:
    """``is_entitled``, expressed as a Chroma ``where`` filter -- the dense push-down
    (Feature 3): match chunks carrying no tags at all (``ACCESS_TAGS_NONE_KEY``, set at
    write time -- see ``vectorstore.ChromaVectorStore._to_stored_metadata``), OR chunks
    whose stored ``access_tags`` list ``$contains`` at least one of the caller's roles.

    Always applied, even when ``auth_context is None`` or has empty ``roles`` -- in both
    cases this reduces to the single untagged-only clause, never "no filter at all"; see
    ``is_entitled``.
    """

    roles = auth_context.roles if auth_context is not None else ()
    clauses: list[dict[str, Any]] = [{ACCESS_TAGS_NONE_KEY: True}]
    clauses.extend({"access_tags": {"$contains": role}} for role in roles if role)
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


class DenseRetriever:
    """Adapter exposing the existing embedder + vector store as a ``Retriever``."""

    strategy = "dense"

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        auth_context: AuthContext | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        with instrumented_step("embed"):
            try:
                query_vector = self._embedder.embed_query(query)
            except Exception as exc:
                raise EmbeddingServiceError(f"Query embedding failed: {exc}") from exc
        # Feature 3: entitlement is pushed into the query itself -- an ineligible chunk
        # is never scored/ranked in the first place, so it can never crowd an eligible
        # chunk out of `top_k`. doc_type is pushed down the same way.
        where = _entitlement_where(auth_context)
        if doc_type is not None:
            where = {"$and": [where, {"doc_type": doc_type}]}

        # Freshness cannot be pushed down: dates are stored as strings and Chroma's
        # range operators only compare numbers. Filtering after a top_k query let stale
        # chunks crowd out fresh ones (a corpus of superseded versions answered "no
        # documents found"), so the query widens until top_k fresh chunks are found or
        # the store has nothing more to return.
        n_results = top_k
        while True:
            try:
                candidates = self._store.query(query_vector, n_results, where=where)
            except Exception as exc:
                raise VectorStoreError(f"Vector store query failed: {exc}") from exc
            fresh = [item for item in candidates if item.chunk.is_fresh()]
            if len(fresh) >= top_k or len(candidates) < n_results:
                break
            try:
                total = self._store.count()
            except Exception as exc:
                raise VectorStoreError(f"Vector store count failed: {exc}") from exc
            if n_results >= total:
                break
            n_results = min(n_results * 4, total)

        ACTIVE_DOCUMENTS.set(len(fresh))
        # Kept as a guard for stores that ignore part of `where`.
        return _filter_by_doc_type(fresh, doc_type)[:top_k]


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], *, k: int = 60
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one, scoring each chunk by ``Σ 1/(k + rank)``.

    Ties break on chunk id so the output is fully deterministic. This score is a
    *ranking* signal only -- ``1/(k+rank)`` values are tiny (~0.016 at best) and not a
    meaningful confidence measure, so callers needing a real confidence score must
    replace it after fusion (see ``_combine_dense_sparse_scores`` / reranking below);
    never compare it against ``RAGEngine``'s confidence threshold directly.
    """

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk_id, item.chunk)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [RetrievedChunk(chunk=chunks[cid], score=score) for cid, score in ordered]


def bm25_confidence(raw_score: float) -> float:
    """Map an unbounded Okapi BM25 score onto ``[0, 1)`` as ``s / (s + 1)``.

    Absolute, not relative to the other candidates: a query whose best match is one
    common word stays low (raw 0.1 -> 0.09, below the default 0.15 confidence
    threshold, so the engine abstains), a rare exact match stays high (raw 8 -> 0.89).
    Monotonic, so it never reorders a BM25 ranking. With the default threshold the
    effective BM25 floor is a raw score of about 0.18.
    """

    if raw_score <= 0.0:
        return 0.0
    return raw_score / (raw_score + 1.0)


def _combine_dense_sparse_scores(
    fused: list[RetrievedChunk],
    dense_ranking: list[RetrievedChunk],
    sparse_ranking: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Replace each fused candidate's raw RRF score with an absolute confidence.

    RRF fuses *rankings*, not scores (see ``reciprocal_rank_fusion``), so its own score
    is useless as a confidence value. Each candidate gets the stronger of its two
    absolute signals: its raw dense cosine similarity and :func:`bm25_confidence` of its
    raw BM25 score (0.0 for a side it is absent from).

    These used to be min-max normalized over the candidate set and averaged 0.5/0.5,
    which made the best candidate score 1.0 (and anything >= 0.5) however poor the
    match -- so hybrid never abstained -- and pinned the worst candidate at 0.0, so it
    was always dropped by the confidence filter. Absolute scores keep the engine's
    threshold meaningful: it abstains when no candidate has real evidence.

    Order is untouched: RRF still decides which chunks are candidates and in what
    order -- only the ``.score`` field is rewritten here.
    """

    dense_scores = {item.chunk.id: item.score for item in dense_ranking}
    sparse_scores = {item.chunk.id: item.score for item in sparse_ranking}
    return [
        RetrievedChunk(
            chunk=item.chunk,
            score=max(
                min(1.0, max(0.0, dense_scores.get(item.chunk.id, 0.0))),
                bm25_confidence(sparse_scores.get(item.chunk.id, 0.0)),
            ),
        )
        for item in fused
    ]


def _cap_per_document(
    items: list[RetrievedChunk], top_k: int, max_per_document: int
) -> list[RetrievedChunk]:
    """Walk ``items`` in rank order, keeping at most ``max_per_document`` chunks per
    parent document, until ``top_k`` are collected (or ``items`` runs out).

    ``max_per_document <= 0`` disables the cap -- same behavior as before this existed.
    """

    if max_per_document <= 0:
        return items[:top_k]

    counts: dict[str, int] = {}
    capped: list[RetrievedChunk] = []
    for item in items:
        doc_id = item.chunk.document_id
        if counts.get(doc_id, 0) >= max_per_document:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        capped.append(item)
        if len(capped) >= top_k:
            break
    return capped


def _apply_reranker(
    reranker: Reranker | None,
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
    max_chunks_per_document: int,
) -> list[RetrievedChunk]:
    """Optionally re-rank ``candidates``, then cap to ``top_k`` per ``_cap_per_document``.

    Shared by ``HybridRetriever`` and ``SparseRetriever`` so the "rerank (degrading to
    the un-reranked ranking on failure) -> per-document cap" sequence lives in one place
    rather than being duplicated per orchestration strategy.
    """

    if reranker is None or not candidates:
        return _cap_per_document(candidates, top_k, max_chunks_per_document)

    with instrumented_step("rerank", candidate_count=len(candidates)) as rec:
        try:
            # Re-ranks the full candidate pool (not just top_k) so the per-document cap
            # below has more than one chunk per document to choose from when picking the
            # final top_k.
            result = reranker.rerank(query, candidates, len(candidates))
        except Exception as exc:
            # Reranking is optional: once its own retries are exhausted, degrade to
            # the un-reranked ranking rather than failing the whole request.
            logger.error(
                "Reranking failed after retries exhausted; skipping rerank: %s", exc
            )
            capped = _cap_per_document(candidates, top_k, max_chunks_per_document)
            rec.set(result_count=len(capped))
            return capped
        rec.set(result_count=len(result))
    return _cap_per_document(result, top_k, max_chunks_per_document)


class SparseRetriever:
    """Orchestration wrapper around a bare lexical retriever (``BM25Retriever``):
    raw scores -> :func:`bm25_confidence` -> optional re-rank -> per-document cap.

    ``retrieval="sparse"`` uses this instead of ``HybridRetriever`` because there is
    nothing to fuse -- BM25's raw Okapi scores are unbounded, not the 0..1-ish confidence
    value ``RAGEngine``'s confidence threshold expects, so they are mapped onto it here,
    per candidate and absolutely (not relative to the other candidates), so the engine
    can still abstain on a weak best match.
    """

    def __init__(
        self,
        sparse: Retriever,
        reranker: Reranker | None = None,
        *,
        candidates: int = 20,
        max_chunks_per_document: int = 2,
    ) -> None:
        self._sparse = sparse
        self._reranker = reranker
        self._candidates = candidates
        self._max_chunks_per_document = max_chunks_per_document

    @property
    def strategy(self) -> str:
        """Descriptive name for observability labels (e.g. metrics, logs)."""

        return "sparse+rerank" if self._reranker is not None else "sparse"

    def refresh(self) -> None:
        """Pass a post-ingest refresh through to the wrapped sparse retriever."""

        refresh_retriever(self._sparse)

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        auth_context: AuthContext | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        # Pull a deeper candidate pool than top_k so re-ranking has room to work, same
        # rationale as HybridRetriever's own `depth`.
        depth = max(self._candidates, top_k)
        ranking = self._sparse.retrieve(
            query, depth, auth_context=auth_context, doc_type=doc_type,
        )
        candidates = [
            RetrievedChunk(chunk=item.chunk, score=bm25_confidence(item.score))
            for item in ranking
        ]

        return _apply_reranker(
            self._reranker, query, candidates, top_k, self._max_chunks_per_document
        )


class HybridRetriever:
    """dense + sparse -> RRF fuse -> top-N candidates -> optional re-rank -> top_k.

    Genuine dense+sparse fusion only -- ``retrieval="sparse"`` (no dense side) is
    ``SparseRetriever``'s job, not this class's. With no ``sparse`` and no ``reranker``
    this behaves exactly like ``DenseRetriever``, which keeps the dense-only baseline
    honest; with no ``sparse`` and a ``reranker`` it is plain "dense -> re-rank".
    """

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever | None = None,
        reranker: Reranker | None = None,
        *,
        rrf_k: int = 60,
        candidates: int = 20,
        max_chunks_per_document: int = 2,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._candidates = candidates
        self._max_chunks_per_document = max_chunks_per_document

    @property
    def strategy(self) -> str:
        """Descriptive name for observability labels (e.g. metrics, logs)."""

        base = "hybrid" if self._sparse is not None else "dense"
        if self._reranker is not None:
            return f"{base}+rerank"
        return base

    def refresh(self) -> None:
        """Pass a post-ingest refresh through to the sparse side, if any.

        The dense side always queries the live vector store, so it has nothing to
        refresh -- only ``self._sparse`` (when this is genuine dense+sparse fusion,
        not the dense-only degenerate case) can hold cached state.
        """

        refresh_retriever(self._sparse)

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        auth_context: AuthContext | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        # Pull a deeper candidate pool than top_k so fusion/re-ranking have room to work.
        depth = max(self._candidates, top_k)
        dense_ranking = self._dense.retrieve(
            query, depth, auth_context=auth_context, doc_type=doc_type,
        )
        rankings = [dense_ranking]
        sparse_ranking: list[RetrievedChunk] | None = None
        if self._sparse is not None:
            sparse_ranking = self._sparse.retrieve(
                query, depth, auth_context=auth_context, doc_type=doc_type,
            )
            rankings.append(sparse_ranking)

        # With a single (dense-only) ranking there is nothing to fuse -- its score is
        # already a meaningful 0..1-ish confidence value (cosine similarity), unlike raw
        # BM25 scores (see `SparseRetriever`, which normalizes those instead).
        if len(rankings) > 1:
            fused = reciprocal_rank_fusion(rankings, k=self._rrf_k)
            # RRF only fuses/orders candidates (Feature 3) -- its own score is not a
            # meaningful confidence value, so replace it with each candidate's absolute
            # dense/BM25 evidence before anything downstream (reranking, the
            # confidence threshold, Answer.score) ever sees it.
            fused = _combine_dense_sparse_scores(fused, dense_ranking, sparse_ranking)
        else:
            fused = rankings[0]
        # Truncate to `depth`, not `candidates`: a caller asking for more than the
        # configured candidate pool must still receive top_k results.
        candidates = fused[:depth]

        return _apply_reranker(
            self._reranker, query, candidates, top_k, self._max_chunks_per_document
        )
