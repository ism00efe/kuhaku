"""Tests for the Vertex AI LLM provider.

No real network calls or Google credentials -- ``google.genai.Client`` is monkeypatched.
"""

from __future__ import annotations

import builtins

import pytest

pytest.importorskip(
    "google.genai",
    reason="Vertex AI provider tests need the optional 'vertex' extra "
    "(pip install 'kuhaku[vertex]'); CI's default install does not include it.",
)

from kuhaku.core.config import Settings
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.base import LLMError


class FakeUsageMetadata:
    def __init__(self, prompt_token_count=None, candidates_token_count=None):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class FakeResponse:
    def __init__(self, text: str | None, usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


class FakeModels:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.last_kwargs: dict | None = None

    def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._response


class FakeClient:
    def __init__(self, models: FakeModels):
        self.models = models


@pytest.fixture
def fake_genai_client(monkeypatch):
    """Patches ``google.genai.Client`` so ``VertexAIProvider.__init__`` never needs real
    credentials, and returns the ``FakeModels`` instance the provider ends up using."""

    import google.genai as genai

    models = FakeModels(response=FakeResponse("hi there "))
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))
    return models


def test_factory_builds_vertex(fake_genai_client):
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    settings = Settings(
        _env_file=None, llm_provider="vertex", vertex_project="p", vertex_location="l"
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, VertexAIProvider)
    assert provider.name == "vertex:gemini-2.5-flash"


def test_generate_returns_text(fake_genai_client):
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(project="p", location="l")
    assert provider.generate("sys", "user") == "hi there"
    assert fake_genai_client.last_kwargs["contents"] == "user"
    assert fake_genai_client.last_kwargs["config"] == {
        "system_instruction": "sys",
        "http_options": {"timeout": 120_000},
    }


def test_generate_populates_last_usage_from_usage_metadata(monkeypatch):
    import google.genai as genai

    response = FakeResponse(
        "hi there",
        usage_metadata=FakeUsageMetadata(prompt_token_count=12, candidates_token_count=5),
    )
    models = FakeModels(response=response)
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))

    from kuhaku.core.llm.base import TokenUsage
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(project="p", location="l")
    assert provider.last_usage is None
    provider.generate("s", "u")
    assert provider.last_usage == TokenUsage(prompt_tokens=12, completion_tokens=5)


def test_generate_without_usage_metadata_leaves_last_usage_none(fake_genai_client):
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(project="p", location="l")
    provider.generate("s", "u")
    assert provider.last_usage is None


def test_generate_with_partial_usage_metadata(monkeypatch):
    import google.genai as genai

    response = FakeResponse(
        "hi there",
        usage_metadata=FakeUsageMetadata(prompt_token_count=12, candidates_token_count=None),
    )
    models = FakeModels(response=response)
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))

    from kuhaku.core.llm.base import TokenUsage
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(project="p", location="l")
    provider.generate("s", "u")
    assert provider.last_usage == TokenUsage(prompt_tokens=12, completion_tokens=None)


def test_resolves_project_and_location_from_env(monkeypatch, fake_genai_client):
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "env-location")
    provider = VertexAIProvider()
    assert provider._project == "env-project"  # noqa: SLF001
    assert provider._location == "env-location"  # noqa: SLF001


def test_constructor_args_take_precedence_over_env(monkeypatch, fake_genai_client):
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "env-location")
    provider = VertexAIProvider(project="explicit-project")
    assert provider._project == "explicit-project"  # noqa: SLF001


def test_sdk_error_becomes_llmerror(monkeypatch):
    import google.genai as genai

    models = FakeModels(error=RuntimeError("quota exceeded"))
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(project="p", location="l")
    with pytest.raises(LLMError):
        provider.generate("s", "u")


def test_empty_response_becomes_llmerror(monkeypatch):
    import google.genai as genai

    models = FakeModels(response=FakeResponse(None))
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(project="p", location="l")
    with pytest.raises(LLMError):
        provider.generate("s", "u")


def test_missing_project_and_location_raises_value_error(monkeypatch):
    import google.genai as genai

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    def _raise(**kwargs):
        raise RuntimeError("Unable to determine project/location")

    monkeypatch.setattr(genai, "Client", _raise)

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    with pytest.raises(ValueError, match="vertex_project"):
        VertexAIProvider(project=None, location=None)


def test_missing_location_only_raises_value_error(monkeypatch):
    import google.genai as genai

    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: calls.append(kwargs))

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    with pytest.raises(ValueError, match="vertex_location"):
        VertexAIProvider(project="p", location=None)
    assert calls == []  # fails before the SDK client is ever constructed


def test_missing_project_only_raises_value_error(monkeypatch):
    import google.genai as genai

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: calls.append(kwargs))

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    with pytest.raises(ValueError, match="vertex_project"):
        VertexAIProvider(project=None, location="l")
    assert calls == []


def test_empty_string_project_and_location_raise_value_error(monkeypatch):
    import google.genai as genai

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr(genai, "Client", lambda **kwargs: calls.append(kwargs))

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    with pytest.raises(ValueError, match="vertex_project"):
        VertexAIProvider(project="", location="")
    assert calls == []


def test_valid_project_and_location_construct_successfully(fake_genai_client):
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(project="p", location="l")
    assert provider._project == "p"  # noqa: SLF001
    assert provider._location == "l"  # noqa: SLF001


def test_client_called_with_vertexai_project_and_location(monkeypatch):
    import google.genai as genai
    from google.genai import types

    captured: dict = {}

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return FakeClient(FakeModels(response=FakeResponse("ok")))

    monkeypatch.setattr(genai, "Client", _fake_client)

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    VertexAIProvider(project="p", location="l")
    assert captured == {
        "vertexai": True,
        "project": "p",
        "location": "l",
        "http_options": types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    }


def _vertex_rag_settings(vertex_project: str = "p", vertex_location: str = "l"):
    """embedding_provider is RAG-specific (lives on RAGSettings, not Settings) -- these
    RAG(...) construction tests need it set to "vertex" so RAG.__init__ builds a
    VertexAIEmbeddings (via the mocked fake_genai_client fixture) instead of trying to
    load a real sentence-transformers model."""

    from kuhaku.tools.rag.config import RAGSettings

    return RAGSettings(
        embedding_provider="vertex", vertex_project=vertex_project, vertex_location=vertex_location
    )


def test_rag_constructor_overrides_vertex_settings(monkeypatch, fake_genai_client):
    from kuhaku import RAG
    from kuhaku.core.config import Settings

    settings = Settings(_env_file=None, llm_provider="vertex")
    rag = RAG(
        settings=settings,
        rag_settings=_vertex_rag_settings(),
        vertex_project="override-project",
        vertex_location="override-location",
    )

    assert rag._settings.vertex_project == "override-project"  # noqa: SLF001
    assert rag._settings.vertex_location == "override-location"  # noqa: SLF001


def test_rag_wraps_provider_with_token_tracking_by_default(fake_genai_client):
    from kuhaku import RAG
    from kuhaku.core.config import Settings
    from kuhaku.core.llm.token_tracking import TokenTrackingLLM

    settings = Settings(
        _env_file=None, llm_provider="vertex", vertex_project="p", vertex_location="l",
    )
    rag = RAG(settings=settings, rag_settings=_vertex_rag_settings())
    assert isinstance(rag._llm, TokenTrackingLLM)  # noqa: SLF001


def test_rag_respects_enable_token_tracking_false(fake_genai_client):
    from kuhaku import RAG
    from kuhaku.core.config import Settings
    from kuhaku.core.llm.token_tracking import TokenTrackingLLM
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    settings = Settings(
        _env_file=None, llm_provider="vertex", vertex_project="p", vertex_location="l",
    )
    rag = RAG(settings=settings, rag_settings=_vertex_rag_settings(), enable_token_tracking=False)
    assert isinstance(rag._llm, VertexAIProvider)  # noqa: SLF001
    assert not isinstance(rag._llm, TokenTrackingLLM)  # noqa: SLF001


def test_rag_respects_enable_token_tracking_true(fake_genai_client):
    from kuhaku import RAG
    from kuhaku.core.config import Settings
    from kuhaku.core.llm.token_tracking import TokenTrackingLLM

    settings = Settings(
        _env_file=None, llm_provider="vertex", vertex_project="p", vertex_location="l",
    )
    rag = RAG(settings=settings, rag_settings=_vertex_rag_settings(), enable_token_tracking=True)
    assert isinstance(rag._llm, TokenTrackingLLM)  # noqa: SLF001


def test_rag_removed_enable_cost_tracking_raises_type_error(fake_genai_client):
    """The 'enable_cost_tracking' deprecation shim was removed; it is now just an
    unrecognized keyword argument like any other, same as `invalid_argument` below."""

    from kuhaku import RAG
    from kuhaku.core.config import Settings

    settings = Settings(
        _env_file=None, llm_provider="vertex", vertex_project="p", vertex_location="l",
    )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        RAG(settings=settings, rag_settings=_vertex_rag_settings(), enable_cost_tracking=False)


def test_rag_unexpected_kwarg_raises_type_error(fake_genai_client):
    from kuhaku import RAG
    from kuhaku.core.config import Settings

    settings = Settings(
        _env_file=None, llm_provider="vertex", vertex_project="p", vertex_location="l",
    )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        RAG(settings=settings, rag_settings=_vertex_rag_settings(), invalid_argument=True)


def test_rag_does_not_double_wrap_already_token_tracked_provider(monkeypatch, fake_genai_client):
    import kuhaku
    from kuhaku.core.config import Settings
    from kuhaku.core.llm.token_tracking import TokenTrackingLLM
    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    settings = Settings(
        _env_file=None, llm_provider="vertex", vertex_project="p", vertex_location="l",
    )
    raw = VertexAIProvider(project="p", location="l")
    already_wrapped = TokenTrackingLLM(raw, provider="vertex")
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s, **_k: already_wrapped)

    rag = kuhaku.RAG(settings=settings, rag_settings=_vertex_rag_settings())
    assert rag._llm is already_wrapped  # noqa: SLF001


# --- RAG construction: Settings vs. RAGSettings (settings-separation) --------------
def test_rag_derives_rag_settings_from_settings_rag_field(fake_genai_client):
    """No explicit rag_settings -- RAG.__init__ derives one from settings.rag (plus the
    vertex_project/vertex_location/retry_enabled overlay, see RAGSettings.from_settings;
    vertex_project/location must be set on Settings itself, not settings.rag, for that
    overlay to apply)."""

    from kuhaku import RAG
    from kuhaku.core.config import Settings
    from kuhaku.tools.rag.config import RAGSettings

    custom_rag = RAGSettings(embedding_provider="vertex", top_k=11)
    settings = Settings(
        _env_file=None, llm_provider="vertex",
        vertex_project="p", vertex_location="l", rag=custom_rag,
    )

    rag = RAG(settings=settings)

    assert rag._rag_settings.top_k == 11  # noqa: SLF001
    assert rag._rag_settings.embedding_provider == "vertex"  # noqa: SLF001
    assert rag._rag_settings.vertex_project == "p"  # noqa: SLF001 -- overlaid from Settings


def test_rag_uses_explicit_rag_settings_verbatim_when_given(fake_genai_client, tmp_path):
    """An explicitly-passed rag_settings is used as-is -- it is not re-derived from or
    merged with settings.rag. chroma_persist_dir must be pre-set here, since RAG.__init__
    still fills in a fresh temp directory (its usual zero-config behavior) whenever the
    resolved RAGSettings' own chroma_persist_dir is unset -- that is the one case where
    even an explicitly-passed rag_settings is not used completely verbatim."""

    from kuhaku import RAG
    from kuhaku.core.config import Settings
    from kuhaku.tools.rag.config import RAGSettings

    settings = Settings(
        _env_file=None, llm_provider="vertex", vertex_project="p", vertex_location="l",
    )
    explicit_rag = RAGSettings(
        embedding_provider="vertex", vertex_project="explicit-p", vertex_location="explicit-l",
        chroma_persist_dir=str(tmp_path),
    )

    rag = RAG(settings=settings, rag_settings=explicit_rag)

    assert rag._rag_settings is explicit_rag  # noqa: SLF001


def test_rag_convenience_kwargs_override_explicit_rag_settings(fake_genai_client):
    """The chunking/embedding/vector_store/corpus_dir convenience kwargs still apply as
    overrides on top of an explicitly-passed rag_settings."""

    from kuhaku import RAG
    from kuhaku.core.config import Settings

    settings = Settings(
        _env_file=None, llm_provider="vertex", vertex_project="p", vertex_location="l",
    )
    explicit_rag = _vertex_rag_settings()

    rag = RAG(settings=settings, rag_settings=explicit_rag, chunking="structural")

    assert rag._rag_settings.chunking_strategy == "structural"  # noqa: SLF001
    assert rag._rag_settings is not explicit_rag  # noqa: SLF001 -- a new instance
    # unrelated fields preserved
    assert rag._rag_settings.embedding_provider == "vertex"  # noqa: SLF001


# --- Retry + circuit breaker ---------------------------------------------------
def test_generate_retries_transient_server_error_then_succeeds(monkeypatch):
    import google.genai as genai
    from google.genai import errors

    calls = {"n": 0}

    def flaky_generate_content(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise errors.ServerError(503, {"error": {"message": "unavailable"}})
        return FakeResponse("hi there")

    models = FakeModels()
    models.generate_content = flaky_generate_content
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(
        project="p", location="l", retry_max_attempts=3,
        retry_backoff_base_seconds=0.001, retry_backoff_max_seconds=0.01,
    )
    assert provider.generate("s", "u") == "hi there"
    assert calls["n"] == 2


def test_generate_does_not_retry_client_error(monkeypatch):
    import google.genai as genai
    from google.genai import errors

    calls = {"n": 0}

    def failing_generate_content(**kwargs):
        calls["n"] += 1
        raise errors.ClientError(400, {"error": {"message": "bad request"}})

    models = FakeModels()
    models.generate_content = failing_generate_content
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(
        project="p", location="l", retry_max_attempts=3,
        retry_backoff_base_seconds=0.001, retry_backoff_max_seconds=0.01,
    )
    with pytest.raises(LLMError):
        provider.generate("s", "u")
    assert calls["n"] == 1  # 4xx is not retried


def test_circuit_opens_after_threshold_and_rejects_without_calling(monkeypatch):
    import google.genai as genai

    models = FakeModels(error=RuntimeError("down"))
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient(models))

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    provider = VertexAIProvider(
        project="p", location="l",
        retry_enabled=False,
        circuit_breaker_failure_threshold=1,
        circuit_breaker_reset_timeout_seconds=60.0,
    )

    with pytest.raises(LLMError):
        provider.generate("s", "u")

    calls_before = models.last_kwargs
    with pytest.raises(LLMError):
        provider.generate("s", "u")
    assert models.last_kwargs is calls_before  # generate_content never called again


def test_missing_google_genai_raises_runtime_error(monkeypatch):
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "google.genai" or name == "google":
            raise ImportError("no module named google.genai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    with pytest.raises(RuntimeError, match="pip install google-genai"):
        VertexAIProvider()
