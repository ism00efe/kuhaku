"""Tests for the Vertex AI embedding provider.

No real network calls or Google credentials -- ``google.genai.Client`` is monkeypatched.
"""

from __future__ import annotations

import builtins

import pytest

from kuhaku.tools.rag.config import RAGSettings
from kuhaku.tools.rag.embeddings import EmbeddingServiceError, build_embedding_provider


class FakeEmbedding:
    def __init__(self, values: list[float]):
        self.values = values


class FakeEmbedResponse:
    def __init__(self, embeddings: list[FakeEmbedding]):
        self.embeddings = embeddings


class FakeModels:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.last_kwargs: dict | None = None

    def embed_content(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._response


class FakeClient:
    def __init__(self, models: FakeModels):
        self.models = models


@pytest.fixture
def fake_genai_client(monkeypatch):
    import google.genai as genai

    models = FakeModels(
        response=FakeEmbedResponse([FakeEmbedding([1.0, 2.0, 3.0])])
    )
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))
    return models


def test_factory_builds_vertex_embeddings(fake_genai_client):
    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    settings = RAGSettings(embedding_provider="vertex", vertex_project="p")
    embedder = build_embedding_provider(settings)
    assert isinstance(embedder, VertexAIEmbeddings)


def test_factory_defaults_to_sentence_transformer(monkeypatch):
    from kuhaku.tools.rag.embeddings import SentenceTransformerEmbeddings

    class FakeST:
        def __init__(self, model_name, device=None):
            pass

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)

    settings = RAGSettings()
    embedder = build_embedding_provider(settings)
    assert isinstance(embedder, SentenceTransformerEmbeddings)


def test_factory_unknown_provider_raises():
    settings = RAGSettings(embedding_provider="does-not-exist")
    with pytest.raises(EmbeddingServiceError):
        build_embedding_provider(settings)


def test_embed_documents_uses_retrieval_document_task_type(fake_genai_client):
    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    embedder = VertexAIEmbeddings(project="p", location="l")
    result = embedder.embed_documents(["hello", "world"])

    assert result == [[1.0, 2.0, 3.0]]
    assert fake_genai_client.last_kwargs["contents"] == ["hello", "world"]
    assert fake_genai_client.last_kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_embed_query_uses_retrieval_query_task_type(fake_genai_client):
    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    embedder = VertexAIEmbeddings(project="p", location="l")
    result = embedder.embed_query("hello")

    assert result == [1.0, 2.0, 3.0]
    assert isinstance(result, list) and all(isinstance(v, float) for v in result)
    assert fake_genai_client.last_kwargs["config"].task_type == "RETRIEVAL_QUERY"


def test_sdk_error_becomes_embedding_service_error(monkeypatch):
    import google.genai as genai

    models = FakeModels(error=RuntimeError("quota exceeded"))
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))

    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    embedder = VertexAIEmbeddings(project="p", location="l")
    with pytest.raises(EmbeddingServiceError):
        embedder.embed_query("hello")


def test_missing_project_and_location_raises_value_error(monkeypatch):
    import google.genai as genai

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    def _raise(**kwargs):
        raise RuntimeError("Unable to determine project/location")

    monkeypatch.setattr(genai, "Client", _raise)

    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    with pytest.raises(ValueError, match="vertex_project"):
        VertexAIEmbeddings(project=None, location=None)


def test_missing_location_only_raises_value_error(monkeypatch):
    import google.genai as genai

    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: calls.append(kwargs))

    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    with pytest.raises(ValueError, match="vertex_location"):
        VertexAIEmbeddings(project="p", location=None)
    assert calls == []  # fails before the SDK client is ever constructed


def test_missing_project_only_raises_value_error(monkeypatch):
    import google.genai as genai

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: calls.append(kwargs))

    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    with pytest.raises(ValueError, match="vertex_project"):
        VertexAIEmbeddings(project=None, location="l")
    assert calls == []


def test_empty_string_project_and_location_raise_value_error(monkeypatch):
    import google.genai as genai

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: calls.append(kwargs))

    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    with pytest.raises(ValueError, match="vertex_project"):
        VertexAIEmbeddings(project="", location="")
    assert calls == []


def test_client_called_with_vertexai_project_and_location(monkeypatch):
    import google.genai as genai

    captured: dict = {}

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return FakeClient(FakeModels())

    monkeypatch.setattr(genai, "Client", _fake_client)

    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    VertexAIEmbeddings(project="p", location="l")
    assert captured == {"vertexai": True, "project": "p", "location": "l"}


def test_missing_google_genai_raises_runtime_error(monkeypatch):
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "google":
            raise ImportError("no module named google.genai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    from kuhaku.tools.rag.vertex_embeddings import VertexAIEmbeddings

    with pytest.raises(RuntimeError, match="pip install google-genai"):
        VertexAIEmbeddings()
