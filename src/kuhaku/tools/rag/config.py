"""RAG-specific configuration.

``RAGSettings`` owns every knob the RAG tool cares about (corpus/chunking/retrieval/
re-ranking/caching/contradiction-detection/query-rewriting, the vector store, and the
RAG-scoped retry sites) -- excluding generic, cross-tool concerns (LLM provider, auth,
the HTTP API, logging, ...) that stay on :class:`~kuhaku.core.config.Settings`.

Every field has a sensible default, so ``RAGSettings()`` is usable standalone -- no
``Settings`` instance required. ``Settings.rag`` (a nested field, see ``core/config.py``)
is the kuhaku-wide default instance, populated from the environment via
``KUHAKU_RAG__<FIELD>``-prefixed variables (e.g. ``KUHAKU_RAG__TOP_K=8``); :meth:`RAGSettings.from_settings`
additionally overlays the handful of fields that stay authoritative on ``Settings`` itself
(``vertex_project``, ``vertex_location``, the ``retry_enabled`` master switch) for callers
that only have a full ``Settings`` instance to hand.
"""

from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import TYPE_CHECKING

from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from kuhaku.core.config import Settings


@dataclass(frozen=True)
class RAGSettings:
    """All RAG-specific configuration, with standalone-usable defaults."""

    # --- Corpus & retrieval -----------------------------------------------------
    corpus_dir: str = ""
    doc_type_prefix_mapping: dict[str, str] = field(default_factory=dict)
    chunk_size: int = 500
    chunk_overlap: int = 80
    chunking_strategy: str = "paragraph"  # paragraph | structural
    top_k: int = 4
    rag_confidence_threshold: float = 0.15
    max_chunks_per_document: int = 2

    # --- Upload limits ------------------------------------------------------------
    # Maximum allowed size, in bytes, of a single uploaded document accepted by
    # extract_text()/ingest_single_document(). None (the default) means no limit is
    # enforced -- application-specific, hence living here rather than on the generic
    # Settings class (see that class's module docstring).
    max_upload_bytes: int | None = None

    # --- Vector store ------------------------------------------------------------
    chroma_persist_dir: str = ""
    chroma_collection: str = "default_kb"
    collection_base: str = "corpus"

    # --- Embeddings ----------------------------------------------------------------
    embedding_provider: str = "sentence-transformer"  # sentence-transformer | vertex
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"
    embedding_model_version: str = "intfloat/multilingual-e5-small"
    vertex_embedding_model: str = "gemini-embedding-001"
    vertex_embedding_dimensions: int = 768
    # Cross-tool platform fields that stay authoritative on Settings (Google Cloud
    # project/location apply beyond just RAG) -- optional here so a RAGSettings built
    # standalone (no Settings instance) can still drive the vertex embedding provider.
    # RAGSettings.from_settings() overlays these from Settings itself.
    vertex_project: str | None = None
    vertex_location: str | None = "us-central1"

    # --- Hybrid retrieval (dense + sparse BM25, fused with RRF) ------------------
    hybrid_enabled: bool = True
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # --- Cross-encoder re-ranking (must be multilingual) --------------------------
    rerank_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 20

    # --- Query rewriting (D48) ------------------------------------------------------
    query_rewrite_enabled: bool = False
    query_rewrite_cache_ttl_seconds: int = 3600

    # --- Contradiction detection (D50) -----------------------------------------------
    contradiction_detection_enabled: bool = True
    contradiction_similarity_threshold: float = 0.75

    # --- Query-answer cache (SQLite) --------------------------------------------------
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # --- Model and prompt versioning (D42) --------------------------------------------
    prod_prompt_version: str = "v2"
    eval_prompt_version: str = "v2"

    # --- Retry (D40): master switch + the RAG-owned retry sites (embedding,
    # vector store, reranker). The LLM retry site (retry_llm_*) is a generic,
    # cross-tool concern and stays on Settings. ---------------------------------------
    retry_enabled: bool = True
    retry_embedding_max_attempts: int = 2
    retry_embedding_backoff_base_seconds: float = 0.5
    retry_embedding_backoff_max_seconds: float = 5.0
    retry_vectorstore_max_attempts: int = 2
    retry_vectorstore_backoff_seconds: float = 0.5
    retry_reranker_max_attempts: int = 2
    retry_reranker_backoff_base_seconds: float = 1.0
    retry_reranker_backoff_max_seconds: float = 5.0

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGSettings:
        """Project the RAG-relevant settings out of a full :class:`Settings` instance.

        ``settings.rag`` already carries every RAG-owned field (populated from defaults
        and/or ``KUHAKU_RAG__*`` environment variables); this overlays the three fields that
        remain authoritative on ``Settings`` itself -- ``vertex_project``,
        ``vertex_location`` (generic Google Cloud platform settings, also needed by the
        Vertex embedding provider) and ``retry_enabled`` (the cross-subsystem master
        retry switch) -- so a caller holding only ``settings.rag`` in isolation would not
        see a ``Settings``-level override of any of those three.
        """

        return dataclasses.replace(
            settings.rag,
            vertex_project=settings.vertex_project,
            vertex_location=settings.vertex_location,
            retry_enabled=settings.retry_enabled,
        )
