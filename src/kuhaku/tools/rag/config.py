"""RAG-specific configuration.

``RAGSettings`` owns every knob the RAG tool cares about (corpus/chunking/retrieval/
re-ranking/caching/contradiction-detection, the vector store, and the RAG-scoped retry
sites) -- excluding generic, cross-tool concerns (LLM provider, auth, the HTTP API,
logging, ...) that stay on :class:`~kuhaku.core.config.Settings`.

Every field has a sensible default, so ``RAGSettings()`` is usable standalone -- no
``Settings`` instance required, and nothing read from the environment.
:meth:`RAGSettings.from_env` reads the tool's own ``KUHAKU_RAG__<FIELD>`` variables
(``KUHAKU_RAG__TOP_K=8``, ``KUHAKU_RAG__DOC_TYPE_PREFIX_MAPPING='{"log_": "x"}'``);
:meth:`RAGSettings.from_settings` is ``from_env()`` plus the fields that stay
authoritative on core's ``Settings`` (``vertex_project``, ``vertex_location``, the
``retry_enabled`` master switch) and is what ``RAG()`` builds when not handed a
``rag_settings``.

This object used to be nested on core's ``Settings`` as ``Settings.rag``. It is a
tool-owned settings object now, like every tool's: ``kuhaku.core`` may never import a
tool, so the tool reads its own environment. The variable names did not change.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import field
from typing import TYPE_CHECKING, Any

from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from kuhaku.core.config import Settings

#: Environment-variable prefix for RAG settings -- the names that worked while these
#: settings were nested on ``Settings`` (``KUHAKU_`` + ``RAG`` + the ``__`` delimiter).
RAG_ENV_PREFIX = "KUHAKU_RAG__"

# Fields whose environment value is JSON rather than a scalar.
_JSON_FIELDS = frozenset({"doc_type_prefix_mapping"})


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

    # --- Embeddings ----------------------------------------------------------------
    embedding_provider: str = "sentence-transformer"  # sentence-transformer | vertex
    embedding_model: str = "intfloat/multilingual-e5-small"
    # "auto" (the default) picks cuda / mps / cpu from what torch reports at build time
    # (build_embedding_provider -> kuhaku.core.resolve.probes.torch_accelerator, which
    # does not go stale). A concrete value is absolute. RAG() passes the device it
    # resolved; a standalone build_embedding_provider() call resolves it itself.
    embedding_device: str = "auto"  # auto | cpu | cuda | mps
    embedding_model_version: str = "intfloat/multilingual-e5-small"
    vertex_embedding_model: str = "gemini-embedding-001"
    vertex_embedding_dimensions: int = 768
    # Cross-tool platform fields that stay authoritative on Settings (Google Cloud
    # project/location apply beyond just RAG) -- optional here so a RAGSettings built
    # standalone (no Settings instance) can still drive the vertex embedding provider.
    # RAGSettings.from_settings() overlays these from Settings itself.
    vertex_project: str | None = None
    vertex_location: str | None = "us-central1"

    # --- Retrieval strategy + hybrid fusion (dense + sparse BM25, fused with RRF) ----
    # "dense" | "sparse" | "hybrid" -- the same vocabulary as kuhaku.RAG's `retrieval`
    # kwarg / `_VALID_RETRIEVAL_MODES` (validated there, at RAG() construction, not
    # here -- see RAG.__init__). "auto" (the default) resolves to "hybrid" when a real
    # embedding provider can be built at construction time, and to "sparse" (BM25 only --
    # no embeddings, no torch, no model download) when it cannot, announcing the
    # downgrade on the terminal. "hybrid" is the baseline: per the project's "a default
    # may cost CPU/memory, never a download" rule, hybrid needs neither (BM25 is built
    # from the vector store's own chunks) and dense alone is weak on exact-match/
    # rare-term queries that BM25 covers. An explicit "dense"/"hybrid"/"sparse" is
    # absolute; KUHAKU_AUTO=false pins "auto" to "hybrid".
    retrieval: str = "auto"  # auto | dense | sparse | hybrid
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # --- Cross-encoder re-ranking (must be multilingual) --------------------------
    # False by default: reranker_model is ~1GB to download plus VRAM, and no default may
    # trigger that on a bare RAG()/`pip install kuhaku` (the same CPU/memory-only rule
    # above). Set true -- or KUHAKU_RAG__RERANK_ENABLED=true -- to opt in from the
    # environment without touching code.
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 20

    # --- Contradiction detection ------------------------------------------------------
    contradiction_detection_enabled: bool = True

    # --- Capability-resolution tunables (§12). Both are estimates, not measurements --
    # marked as such in the code that reads them (store_policy.CHUNK_UPGRADE_THRESHOLD,
    # hardware.VRAM_HEADROOM), whose module-level constants supply these defaults.
    # chunk_upgrade_threshold: RAG.ingest()/load_documents() suggest a heavier store once
    # the store holds more chunks than this. vram_headroom: the fraction of VRAM RAG
    # reserves beyond model weights when it reports which model size class should fit.
    chunk_upgrade_threshold: int = 50_000
    vram_headroom: float = 0.25

    # --- Query-answer cache (SQLite) --------------------------------------------------
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    # Mirrors Settings.audit_log_path's own convention: "" (the default) means "use the
    # kuhaku-managed default location" (./data/kuhaku_qa_cache.sqlite3, created if
    # missing -- see rag/cache.py's _DEFAULT_CACHE_DB_PATH), not "caching disabled" --
    # that's what cache_enabled is for.
    cache_db_path: str = ""

    # --- Model and prompt versioning ---------------------------------------------------
    prod_prompt_version: str = "v2"
    eval_prompt_version: str = "v2"

    # --- Retry: master switch + the RAG-owned retry sites (embedding, vector store,
    # reranker). The LLM retry site (retry_llm_*) is a generic, cross-tool concern and
    # stays on Settings. ------------------------------------------------------------
    retry_enabled: bool = True
    retry_embedding_max_attempts: int = 2
    retry_embedding_backoff_base_seconds: float = 0.5
    retry_embedding_backoff_max_seconds: float = 5.0
    retry_vectorstore_max_attempts: int = 2
    retry_vectorstore_backoff_seconds: float = 0.5
    retry_reranker_max_attempts: int = 2
    retry_reranker_backoff_base_seconds: float = 1.0
    retry_reranker_backoff_max_seconds: float = 5.0

    def __post_init__(self) -> None:
        # chunk_overlap >= chunk_size used to be accepted and silently clamped to a
        # 1-character stride when a long paragraph was hard-split: thousands of
        # near-identical chunks from one paragraph.
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size; got "
                f"chunk_overlap={self.chunk_overlap}, chunk_size={self.chunk_size}"
            )

    @classmethod
    def from_env(cls, env_file: str | None = None, **overrides: Any) -> RAGSettings:
        """Build from ``KUHAKU_RAG__<FIELD>`` variables (case-insensitive), then
        ``overrides``.

        ``env_file``, when given, is a dotenv file read first; the process environment
        wins over it, as for ``Settings``. Values are coerced to each field's type
        (``"8"`` -> ``8``, ``"true"`` -> ``True``); a mapping field takes JSON. A bad
        value raises; an unknown ``KUHAKU_RAG__`` name is ignored -- both as when these
        variables were read through ``Settings``.
        """

        raw: dict[str, str] = {}
        if env_file is not None:
            from dotenv import dotenv_values

            raw.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
        raw.update(os.environ)

        names = {f.name for f in dataclasses.fields(cls)}
        prefix = RAG_ENV_PREFIX.lower()
        values: dict[str, Any] = {}
        for key, value in raw.items():
            if not key.lower().startswith(prefix):
                continue
            name = key[len(prefix):].lower()
            if name in names:
                values[name] = json.loads(value) if name in _JSON_FIELDS else value
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_settings(cls, settings: Settings, *, env_file: str | None = None) -> RAGSettings:
        """:meth:`from_env` with the three fields that remain authoritative on
        :class:`Settings` overlaid from ``settings`` -- ``vertex_project``,
        ``vertex_location`` (generic Google Cloud platform settings, also needed by the
        Vertex embedding provider) and ``retry_enabled`` (the cross-subsystem master
        retry switch)."""

        return cls.from_env(
            env_file,
            vertex_project=settings.vertex_project,
            vertex_location=settings.vertex_location,
            retry_enabled=settings.retry_enabled,
        )
