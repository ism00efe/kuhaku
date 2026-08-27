"""Tests for RAG-specific settings: RAGSettings itself and Settings.rag nesting."""

from __future__ import annotations

import dataclasses

from kuhaku.core.config import Settings
from kuhaku.tools.rag.config import RAGSettings

# --- RAGSettings: standalone defaults --------------------------------------------

def test_ragsettings_instantiates_with_no_arguments():
    """RAGSettings must be usable standalone -- no Settings instance required."""

    settings = RAGSettings()
    assert settings.top_k == 4
    assert settings.chunk_size == 500
    assert settings.retrieval == "auto"
    assert settings.rerank_enabled is False
    assert settings.embedding_model == "intfloat/multilingual-e5-small"


def test_ragsettings_defaults_cover_the_fields_moved_from_settings():
    """The 11 fields the settings-separation task added to RAGSettings (versioning +
    the RAG-owned retry sites), plus the vertex/retry_enabled cross-tool fields, all
    have sensible standalone defaults matching Settings' former defaults."""

    settings = RAGSettings()
    assert settings.embedding_model_version == "intfloat/multilingual-e5-small"
    assert settings.prod_prompt_version == "v2"
    assert settings.eval_prompt_version == "v2"
    assert settings.retry_embedding_max_attempts == 2
    assert settings.retry_embedding_backoff_base_seconds == 0.5
    assert settings.retry_embedding_backoff_max_seconds == 5.0
    assert settings.retry_vectorstore_max_attempts == 2
    assert settings.retry_vectorstore_backoff_seconds == 0.5
    assert settings.retry_reranker_max_attempts == 2
    assert settings.retry_reranker_backoff_base_seconds == 1.0
    assert settings.retry_reranker_backoff_max_seconds == 5.0
    assert settings.retry_enabled is True
    assert settings.vertex_project is None
    assert settings.vertex_location == "us-central1"


def test_ragsettings_max_upload_bytes_defaults_to_none():
    """None means no upload size limit is enforced (see ingestion.py's extract_text /
    ingest_single_document)."""

    assert RAGSettings().max_upload_bytes is None


def test_ragsettings_max_upload_bytes_accepts_override():
    assert RAGSettings(max_upload_bytes=1_000_000).max_upload_bytes == 1_000_000


def test_settings_has_no_max_upload_bytes_field():
    """max_upload_bytes is application-specific (upload size is only meaningful in the
    context of a file-upload feature) and moved to RAGSettings; Settings must not carry
    it."""

    assert not hasattr(Settings(_env_file=None), "max_upload_bytes")


def test_ragsettings_accepts_overrides_at_construction():
    settings = RAGSettings(top_k=8, chunk_size=777, rerank_enabled=True)
    assert settings.top_k == 8
    assert settings.chunk_size == 777
    assert settings.rerank_enabled is True


def test_ragsettings_is_frozen():
    settings = RAGSettings()
    try:
        settings.top_k = 99  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("RAGSettings should be immutable")


# --- RAGSettings.from_settings(): mapping completeness ---------------------------

def test_from_settings_passes_through_nested_rag_fields():
    settings = Settings(_env_file=None, rag=RAGSettings(top_k=8, chunk_size=777))
    rag_settings = RAGSettings.from_settings(settings)
    assert rag_settings.top_k == 8
    assert rag_settings.chunk_size == 777


def test_from_settings_reflects_defaults():
    settings = Settings(_env_file=None)
    rag_settings = RAGSettings.from_settings(settings)
    assert rag_settings.chunking_strategy == settings.rag.chunking_strategy
    assert rag_settings.embedding_model == settings.rag.embedding_model
    assert rag_settings.chroma_collection == settings.rag.chroma_collection


def test_from_settings_overlays_vertex_project_and_location_from_settings():
    """vertex_project/vertex_location stay authoritative on Settings (generic Google
    Cloud platform settings) -- from_settings() overlays them onto the derived
    RAGSettings, since KUHAKU_RAG__* nested env vars have no way to reach them."""

    settings = Settings(_env_file=None, vertex_project="p", vertex_location="l")
    rag_settings = RAGSettings.from_settings(settings)
    assert rag_settings.vertex_project == "p"
    assert rag_settings.vertex_location == "l"


def test_from_settings_overlays_retry_enabled_master_switch():
    settings = Settings(_env_file=None, retry_enabled=False)
    rag_settings = RAGSettings.from_settings(settings)
    assert rag_settings.retry_enabled is False


def test_from_settings_passes_through_every_field_not_overlaid_from_settings():
    """Every RAGSettings field except vertex_project/vertex_location/retry_enabled
    (deliberately overlaid from Settings itself, see above) must come through unchanged
    from settings.rag -- guards against a future field silently getting dropped or
    reset if from_settings() is ever rewritten as manual field-by-field construction."""

    custom_rag = RAGSettings(
        top_k=11, chunk_size=999, chunking_strategy="structural",
        embedding_model="custom-model", rerank_enabled=False,
        cache_ttl_seconds=42, retry_embedding_max_attempts=9,
    )
    settings = Settings(_env_file=None, rag=custom_rag)
    derived = RAGSettings.from_settings(settings)

    overlaid = {"vertex_project", "vertex_location", "retry_enabled"}
    for f in dataclasses.fields(RAGSettings):
        if f.name in overlaid:
            continue
        assert getattr(derived, f.name) == getattr(custom_rag, f.name), f.name


def test_ragsettings_from_settings_result_is_frozen():
    settings = Settings(_env_file=None)
    rag_settings = RAGSettings.from_settings(settings)
    try:
        rag_settings.top_k = 99  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("RAGSettings should be immutable")


# --- Settings.rag: nested configuration loading -----------------------------------

def test_settings_rag_defaults_to_a_ragsettings_instance():
    settings = Settings(_env_file=None)
    assert isinstance(settings.rag, RAGSettings)
    assert settings.rag == RAGSettings()


def test_settings_rag_accepts_a_ragsettings_instance_directly():
    settings = Settings(_env_file=None, rag=RAGSettings(top_k=11))
    assert settings.rag.top_k == 11


def test_settings_rag_loads_scalar_fields_from_nested_env_vars(monkeypatch):
    monkeypatch.setenv("KUHAKU_RAG__TOP_K", "9")
    monkeypatch.setenv("KUHAKU_RAG__RERANK_ENABLED", "false")
    settings = Settings(_env_file=None)
    assert settings.rag.top_k == 9
    assert settings.rag.rerank_enabled is False


def test_settings_rag_loads_retrieval_from_nested_env_var(monkeypatch):
    """KUHAKU_RAG__RETRIEVAL is a plain string field here -- RAGSettings itself does not
    validate it against dense/sparse/hybrid (RAG.__init__ is the one place that does,
    the same way for an explicit argument or a settings-derived value -- see
    kuhaku/__init__.py)."""

    monkeypatch.setenv("KUHAKU_RAG__RETRIEVAL", "sparse")
    settings = Settings(_env_file=None)
    assert settings.rag.retrieval == "sparse"


def test_settings_rag_loads_dict_field_from_nested_env_var(monkeypatch):
    monkeypatch.setenv("KUHAKU_RAG__DOC_TYPE_PREFIX_MAPPING", '{"log_": "log_sample"}')
    settings = Settings(_env_file=None)
    assert settings.rag.doc_type_prefix_mapping == {"log_": "log_sample"}


def test_settings_rag_nested_env_vars_do_not_affect_generic_fields(monkeypatch):
    """The "__" nested delimiter must not leak into ordinary flat env var parsing --
    no top-level Settings field name contains "__"."""

    monkeypatch.setenv("KUHAKU_RAG__TOP_K", "9")
    monkeypatch.setenv("KUHAKU_LLM_PROVIDER", "anthropic")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "anthropic"
    assert settings.rag.top_k == 9


# --- Settings must stay generic ---------------------------------------------------

def test_settings_has_no_rag_specific_fields():
    """RAG fields must live only on Settings.rag, never directly on Settings -- adding a
    second tool to kuhaku must never require touching this class."""

    settings = Settings(_env_file=None)
    rag_only_field_names = {f.name for f in dataclasses.fields(RAGSettings)} - {
        # Cross-tool fields RAGSettings mirrors but that legitimately stay on Settings
        # too (see RAGSettings.from_settings()).
        "vertex_project", "vertex_location", "retry_enabled",
    }
    for name in rag_only_field_names:
        assert not hasattr(settings, name), f"Settings should not carry RAG field {name!r}"
