"""Local embedding provider.

Programmed against a small ``EmbeddingProvider`` protocol so the rest of the system does
not depend on sentence-transformers directly. The default implementation runs a local
multilingual model (E5), which is required because the knowledge base is English while
user questions are Turkish (cross-lingual retrieval).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tenacity import wait_exponential

from kuhaku.core.retry import call_with_retry

if TYPE_CHECKING:
    from kuhaku.tools.rag.config import RAGSettings

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """The embedding backend failed (model load error, OOM, or similar).

    Raised by :class:`~kuhaku.tools.rag.retriever.DenseRetriever` when the
    injected :class:`EmbeddingProvider` raises — see DECISIONS.md D33.
    """


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors. Documents and queries are embedded separately because
    some models (E5) expect different prefixes for each."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddings:
    """Local embeddings via ``sentence-transformers``.

    E5 models are trained with ``"query: "`` / ``"passage: "`` prefixes; we add them
    automatically when the model name looks like an E5 model. Vectors are L2-normalized
    so a dot product equals cosine similarity.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        retry_enabled: bool = True,
        retry_max_attempts: int = 2,
        retry_backoff_base_seconds: float = 0.5,
        retry_backoff_max_seconds: float = 5.0,
    ) -> None:
        # Imported lazily so unit tests that don't need embeddings stay fast.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s'...", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._use_e5_prefixes = "e5" in model_name.lower()
        self._retry_enabled = retry_enabled
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_base_seconds = retry_backoff_base_seconds
        self._retry_backoff_max_seconds = retry_backoff_max_seconds

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._use_e5_prefixes:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        # Retry only applies here, not embed_documents/_encode directly -- that method is
        # also used by the ingestion pipeline, which is explicitly out of scope for
        # retry (D40). This is a local in-process computation (no network), so "any
        # exception" is the retryable set here, per D40 -- the motivating transient
        # failure is momentary GPU memory pressure, which has no clean exception type to
        # filter on.
        if self._use_e5_prefixes:
            text = f"query: {text}"
        return call_with_retry(
            lambda: self._encode([text])[0],
            service="embedding",
            enabled=self._retry_enabled,
            max_attempts=self._retry_max_attempts,
            wait=wait_exponential(
                multiplier=self._retry_backoff_base_seconds,
                max=self._retry_backoff_max_seconds,
            ),
        )


def build_embedding_provider(settings: RAGSettings) -> EmbeddingProvider:
    """Instantiate the embedding provider selected by ``settings.embedding_provider``.

    Mirrors ``kuhaku.core.llm.build_llm_provider``. Defaults to the local
    sentence-transformers model (unchanged behavior); ``vertex`` opts into Google Vertex
    AI embeddings for users with Google Cloud credits. Takes a
    :class:`~kuhaku.tools.rag.config.RAGSettings` (not the flat
    ``kuhaku.core.config.Settings``) -- callers holding a full ``Settings`` instance
    should pass ``RAGSettings.from_settings(settings)``.
    """

    provider = settings.embedding_provider.strip().lower()

    if provider == "vertex":
        from .vertex_embeddings import VertexAIEmbeddings

        return VertexAIEmbeddings(
            model=settings.vertex_embedding_model,
            dimensions=settings.vertex_embedding_dimensions,
            project=settings.vertex_project,
            location=settings.vertex_location,
        )

    if provider == "sentence-transformer":
        return SentenceTransformerEmbeddings(
            settings.embedding_model,
            device=settings.embedding_device,
            retry_enabled=settings.retry_enabled,
            retry_max_attempts=settings.retry_embedding_max_attempts,
            retry_backoff_base_seconds=settings.retry_embedding_backoff_base_seconds,
            retry_backoff_max_seconds=settings.retry_embedding_backoff_max_seconds,
        )

    raise EmbeddingServiceError(
        f"Unknown embedding_provider '{settings.embedding_provider}'. "
        "Expected one of: sentence-transformer, vertex."
    )
