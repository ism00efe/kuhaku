"""Google Vertex AI embeddings via the ``google-genai`` SDK.

Optional alternative to :class:`~kuhaku.tools.rag.embeddings.SentenceTransformerEmbeddings`
for users with Google Cloud credits. The SDK is imported lazily so ``google-genai`` stays
an optional dependency (see ``pyproject.toml``'s ``vertex`` extra).
"""

from __future__ import annotations

import logging
import os

from .embeddings import EmbeddingServiceError

logger = logging.getLogger(__name__)

_MISSING_CONFIG_MESSAGE = """Vertex AI configuration is missing. Please set `vertex_project` and `vertex_location` via Settings:

from kuhaku.core.config import get_settings
get_settings().vertex_project = "your-project-id"
get_settings().vertex_location = "us-central1\""""


class VertexAIEmbeddings:
    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Vertex AI support requires: pip install google-genai"
            ) from exc

        self._model = model
        self._dimensions = dimensions
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        if not self._project or not self._location:
            raise ValueError(_MISSING_CONFIG_MESSAGE)
        try:
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )
        except Exception as exc:  # noqa: BLE001 - SDK exception types must not leak
            raise ValueError(_MISSING_CONFIG_MESSAGE) from exc

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai.types import EmbedContentConfig

        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config=EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._dimensions,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - SDK exception types must not leak
            raise EmbeddingServiceError(f"Vertex AI embedding request failed: {exc}") from exc

        try:
            return [list(embedding.values) for embedding in response.embeddings]
        except (AttributeError, TypeError) as exc:
            raise EmbeddingServiceError(
                f"Unexpected Vertex AI embedding response: {response!r}"
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]
