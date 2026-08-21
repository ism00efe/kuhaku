"""Vector store abstraction + ChromaDB implementation.

The pipeline talks to the ``VectorStore`` protocol only, so Chroma can later be swapped
for FAISS/Qdrant without touching ingestion or retrieval. Chunk metadata is stored
alongside each vector so retrieval can produce citations.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

from tenacity import wait_fixed

from kuhaku.tools.rag.models import ACCESS_TAGS_NONE_KEY, Chunk, RetrievedChunk
from kuhaku.core.retry import call_with_retry

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """The vector store backend failed (connection, corruption, or similar).

    Raised by :class:`~kuhaku.tools.rag.retriever.DenseRetriever` and
    :class:`~kuhaku.tools.rag.engine.RAGEngine` when the injected
    :class:`VectorStore` raises — see DECISIONS.md D33.
    """


@runtime_checkable
class VectorStore(Protocol):
    """Minimal interface the RAG pipeline needs from a vector database."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def query(
        self,
        embedding: list[float],
        top_k: int,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """``where``, when given, is a backend-native metadata filter applied *before*
        similarity ranking (not a post-hoc trim of the ranked results) -- for
        :class:`ChromaVectorStore` this is Chroma's own ``where`` filter syntax. Used by
        :class:`~kuhaku.tools.rag.retriever.DenseRetriever` to push document-level access
        filtering down into the query itself (Feature 3: filter before rank). ``None``
        (the default) means no filtering, unchanged from before this parameter existed.
        """
        ...

    def count(self) -> int: ...

    def reset(self) -> None: ...

    def list_collections(self) -> list[str]: ...

    def iter_chunks(self) -> Iterator[Chunk]:
        """Every chunk currently held, in no particular order.

        The single source of truth for "what is actually indexed" -- callers that need
        to rebuild a secondary index (e.g. BM25, see
        ``sparse_retriever.build_bm25_from_store``) from the same chunks the dense side
        already has, instead of re-deriving them from wherever they originally came
        from. A generator rather than a ``list``: the one real consumer needs every
        chunk in memory at once regardless (BM25's corpus-wide statistics require it),
        so this buys that consumer nothing extra -- but it keeps a future consumer that
        only wants to stream through the corpus (e.g. a re-embedding job) from being
        forced to materialize the whole thing too. Implementations must still page
        through their backend rather than issuing one unbounded read (see
        ``ChromaVectorStore.iter_chunks``); an empty store yields nothing rather than
        raising.
        """
        ...


class ChromaVectorStore:
    """Persistent local Chroma collection using cosine distance."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        *,
        retry_enabled: bool = True,
        retry_max_attempts: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._retry_enabled = retry_enabled
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        logger.info(
            "Chroma ready (dir=%s, collection=%s, count=%d)",
            persist_dir,
            collection_name,
            self.count(),
        )

    def _refresh_collection_if_stale(self) -> Any:
        try:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.warning("Failed to refresh collection '%s': %s", self._collection_name, e)
        return self._collection

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        metadatas = [self._to_stored_metadata(c) for c in chunks]
        try:
            self._collection.add(
                ids=[c.id for c in chunks],
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=[c.text for c in chunks],
                metadatas=metadatas,
            )
        except Exception as e:
            if "does not exist" in str(e).lower():
                self._refresh_collection_if_stale().add(
                    ids=[c.id for c in chunks],
                    embeddings=embeddings,  # type: ignore[arg-type]
                    documents=[c.text for c in chunks],
                    metadatas=metadatas,
                )
            else:
                raise

    @staticmethod
    def _to_stored_metadata(chunk: Chunk) -> dict[str, Any]:
        """``Chunk.metadata()`` -> the exact dict written to Chroma.

        Chroma rejects an empty-list metadata value outright (verified against the
        installed client: ``add()`` raises rather than storing ``[]``), so an untagged
        chunk's ``"access_tags"`` key is dropped instead of stored empty; in its place,
        ``ACCESS_TAGS_NONE_KEY`` is set so a `where` filter can still select "carries no
        tags" -- this Chroma version has no ``$exists``-style operator either (also
        verified directly). Never part of ``Chunk.metadata()``'s own shape: purely a
        storage/query-time detail of this backend.
        """

        meta = dict(chunk.metadata())
        tags = meta.pop("access_tags")
        if tags:
            meta["access_tags"] = tags
        else:
            meta[ACCESS_TAGS_NONE_KEY] = True
        return meta

    def query(
        self,
        embedding: list[float],
        top_k: int,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        # Constant backoff (not exponential): disk I/O contention recovers fast, so
        # exponential growth is overkill here (D40). Only .query() is retried -- .add()
        # (ingestion), .count(), .reset() are unaffected.
        try:
            result = call_with_retry(
                lambda: self._collection.query(
                    query_embeddings=[embedding],  # type: ignore[arg-type]
                    n_results=top_k,
                    where=where,  # type: ignore[arg-type]
                    include=["documents", "metadatas", "distances"],
                ),
                service="vectorstore",
                enabled=self._retry_enabled,
                max_attempts=self._retry_max_attempts,
                wait=wait_fixed(self._retry_backoff_seconds),
            )
        except Exception as e:
            if "does not exist" in str(e).lower():
                self._refresh_collection_if_stale()
                result = call_with_retry(
                    lambda: self._collection.query(
                        query_embeddings=[embedding],  # type: ignore[arg-type]
                        n_results=top_k,
                        where=where,  # type: ignore[arg-type]
                        include=["documents", "metadatas", "distances"],
                    ),
                    service="vectorstore",
                    enabled=self._retry_enabled,
                    max_attempts=self._retry_max_attempts,
                    wait=wait_fixed(self._retry_backoff_seconds),
                )
            else:
                raise

        # Chroma's stubs mark these Optional because `include` is dynamic; we always
        # request all three, so they're guaranteed present here.
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]  # type: ignore[index]
        metas = result.get("metadatas", [[]])[0]  # type: ignore[index]
        dists = result.get("distances", [[]])[0]  # type: ignore[index]

        retrieved: list[RetrievedChunk] = []
        for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=True):
            chunk = self._chunk_from_row(cid, text, meta)
            # Chroma returns cosine distance in [0, 2]; convert to a similarity score.
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist)))
        return retrieved

    def get_by_document_id(self, document_id: str) -> list[Chunk]:
        """Every chunk belonging to ``document_id``, in no particular order.

        Unlike :meth:`query`, this is an exact metadata filter, not a similarity search --
        used by the evaluation harness's golden-chunk labeling step (D47), which needs the
        *complete* set of chunks for a known-relevant document, not just the top-k nearest
        neighbours of some query.
        """

        try:
            result = self._collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            if "does not exist" in str(e).lower():
                result = self._refresh_collection_if_stale().get(
                    where={"document_id": document_id},
                    include=["documents", "metadatas"],
                )
            else:
                raise
        ids = result.get("ids", [])
        docs = result.get("documents", []) or []
        metas = result.get("metadatas", []) or []
        return [
            self._chunk_from_row(cid, text, meta)
            for cid, text, meta in zip(ids, docs, metas, strict=True)
        ]

    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Every chunk whose id is in ``chunk_ids``, missing ids silently omitted (D49).

        A debugging utility -- e.g. verifying a D49 replay snapshot's chunks are still
        present in a possibly-rebuilt collection -- not a pipeline dependency (replay
        itself is self-contained via the snapshot's own stored chunk text, never this
        method). Chroma's ``.get(ids=...)`` doesn't guarantee input order or that every
        id exists; same caveat as :meth:`get_by_document_id`.
        """

        if not chunk_ids:
            return []
        logger.debug("looking up %d chunk id(s) by exact id", len(chunk_ids))
        try:
            result = self._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        except Exception as e:
            if "does not exist" in str(e).lower():
                result = self._refresh_collection_if_stale().get(
                    ids=chunk_ids, include=["documents", "metadatas"]
                )
            else:
                raise
        ids = result.get("ids", [])
        docs = result.get("documents", []) or []
        metas = result.get("metadatas", []) or []
        return [
            self._chunk_from_row(cid, text, meta)
            for cid, text, meta in zip(ids, docs, metas, strict=True)
        ]

    def iter_chunks(self, *, page_size: int = 500) -> Iterator[Chunk]:
        """Yield every chunk in the collection, reconstructed via :meth:`_chunk_from_row`.

        Pages through Chroma's ``get(limit=, offset=)`` rather than one unbounded read,
        so a large collection is never pulled into a single response. ``page_size`` is
        the per-request page, not a cap on the total -- paging continues until a page
        comes back short of ``page_size``.
        """

        offset = 0
        while True:
            try:
                result = self._collection.get(
                    include=["documents", "metadatas"], limit=page_size, offset=offset,
                )
            except Exception as e:
                if "does not exist" in str(e).lower():
                    result = self._refresh_collection_if_stale().get(
                        include=["documents", "metadatas"], limit=page_size, offset=offset,
                    )
                else:
                    raise
            ids = result.get("ids", [])
            if not ids:
                return
            docs = result.get("documents", []) or []
            metas = result.get("metadatas", []) or []
            for cid, text, meta in zip(ids, docs, metas, strict=True):
                yield self._chunk_from_row(cid, text, meta)
            if len(ids) < page_size:
                return
            offset += page_size

    @staticmethod
    def _chunk_from_row(cid: str, text: str, meta: Mapping[str, Any]) -> Chunk:
        """Reconstruct a :class:`Chunk` from a stored Chroma row (id/document/metadata).

        Shared by every read path (:meth:`query`, :meth:`get_by_document_id`,
        :meth:`get_by_ids`, :meth:`iter_chunks`) -- all of them read back the same
        metadata shape written by :meth:`add` (``Chunk.metadata()``).
        """

        return Chunk(
            id=cid,
            document_id=str(meta.get("document_id", "")),
            title=str(meta.get("title", "")),
            doc_type=str(meta.get("doc_type", "")),
            text=text,
            chunk_index=int(meta.get("chunk_index", 0)),
            source_path=str(meta.get("source_path", "")),
            # .get(..., "text"): rows written before content_type existed have no
            # such key in stored Chroma metadata -- default keeps old and new
            # chunks coexisting safely in one collection.
            content_type=str(meta.get("content_type", "text")),
            # Same backward-compat rationale for the FR4 freshness fields: rows
            # written before they existed have no such keys.
            effective_date=str(meta.get("effective_date", "")),
            obsolete=bool(meta.get("obsolete", False)),
            expiry_date=str(meta.get("expiry_date", "")),
            # Absent for an untagged chunk (see _to_stored_metadata) and for any row
            # written before this field existed -- both default to "no tags", identical
            # to Chunk's own field default.
            access_tags=tuple(meta.get("access_tags") or ()),
        )

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as e:
            if "does not exist" in str(e).lower():
                return self._refresh_collection_if_stale().count()
            raise

    def reset(self) -> None:
        """Drop and recreate the collection (used before a fresh ingest)."""

        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' reset.", self._collection_name)

    def list_collections(self) -> list[str]:
        collections = self._client.list_collections()
        return [str(getattr(c, "name", c)) for c in collections]
