"""Tests for RAG-specific settings: RAGSettings itself and Settings.rag nesting."""

from __future__ import annotations

import dataclasses

import pytest

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


# --- RAGSettings.from_env()/from_settings() -------------------------------------
# These used to go through a nested `Settings.rag` field, i.e. core importing the RAG
# tool. RAGSettings now reads its own KUHAKU_RAG__* variables; the names are unchanged.


def test_from_settings_reads_rag_fields_from_the_environment(monkeypatch):
    monkeypatch.setenv("KUHAKU_RAG__TOP_K", "8")
    monkeypatch.setenv("KUHAKU_RAG__CHUNK_SIZE", "777")
    rag_settings = RAGSettings.from_settings(Settings(_env_file=None))
    assert rag_settings.top_k == 8
    assert rag_settings.chunk_size == 777


def test_from_settings_reflects_defaults():
    rag_settings = RAGSettings.from_settings(Settings(_env_file=None))
    defaults = RAGSettings()
    assert rag_settings.chunking_strategy == defaults.chunking_strategy
    assert rag_settings.embedding_model == defaults.embedding_model
    assert rag_settings.chroma_collection == defaults.chroma_collection


def test_from_settings_overlays_vertex_project_and_location_from_settings():
    """vertex_project/vertex_location stay authoritative on Settings (generic Google
    Cloud platform settings) -- from_settings() overlays them onto the derived
    RAGSettings."""

    settings = Settings(_env_file=None, vertex_project="p", vertex_location="l")
    rag_settings = RAGSettings.from_settings(settings)
    assert rag_settings.vertex_project == "p"
    assert rag_settings.vertex_location == "l"


def test_from_settings_overlays_retry_enabled_master_switch():
    settings = Settings(_env_file=None, retry_enabled=False)
    rag_settings = RAGSettings.from_settings(settings)
    assert rag_settings.retry_enabled is False


def test_ragsettings_from_settings_result_is_frozen():
    rag_settings = RAGSettings.from_settings(Settings(_env_file=None))
    try:
        rag_settings.top_k = 99  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("RAGSettings should be immutable")


def test_from_env_defaults_to_ragsettings_defaults():
    assert RAGSettings.from_env() == RAGSettings()


def test_from_env_loads_scalar_fields(monkeypatch):
    monkeypatch.setenv("KUHAKU_RAG__TOP_K", "9")
    monkeypatch.setenv("KUHAKU_RAG__RERANK_ENABLED", "false")
    rag_settings = RAGSettings.from_env()
    assert rag_settings.top_k == 9
    assert rag_settings.rerank_enabled is False


def test_from_env_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("kuhaku_rag__top_k", "7")
    assert RAGSettings.from_env().top_k == 7


def test_from_env_loads_retrieval(monkeypatch):
    """A plain string here -- RAG.__init__ is the one place that validates it."""

    monkeypatch.setenv("KUHAKU_RAG__RETRIEVAL", "sparse")
    assert RAGSettings.from_env().retrieval == "sparse"


def test_from_env_loads_dict_field_as_json(monkeypatch):
    monkeypatch.setenv("KUHAKU_RAG__DOC_TYPE_PREFIX_MAPPING", '{"log_": "log_sample"}')
    assert RAGSettings.from_env().doc_type_prefix_mapping == {"log_": "log_sample"}


def test_from_env_reads_a_dotenv_file_and_the_environment_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("KUHAKU_RAG__TOP_K=5\nKUHAKU_RAG__CHUNK_SIZE=600\n", encoding="utf-8")
    monkeypatch.setenv("KUHAKU_RAG__TOP_K", "6")
    rag_settings = RAGSettings.from_env(str(env_file))
    assert rag_settings.top_k == 6
    assert rag_settings.chunk_size == 600


def test_from_env_rejects_a_bad_value_and_ignores_unknown_names(monkeypatch):
    monkeypatch.setenv("KUHAKU_RAG__NOT_A_FIELD", "x")
    assert RAGSettings.from_env() == RAGSettings()
    monkeypatch.setenv("KUHAKU_RAG__TOP_K", "many")
    with pytest.raises(ValueError):
        RAGSettings.from_env()


def test_from_env_overrides_win(monkeypatch):
    monkeypatch.setenv("KUHAKU_RAG__TOP_K", "9")
    assert RAGSettings.from_env(top_k=3).top_k == 3


def test_rag_env_vars_do_not_affect_generic_fields(monkeypatch):
    monkeypatch.setenv("KUHAKU_RAG__TOP_K", "9")
    monkeypatch.setenv("KUHAKU_LLM_PROVIDER", "anthropic")
    assert Settings(_env_file=None).llm_provider == "anthropic"
    assert RAGSettings.from_env().top_k == 9


def test_settings_rag_keyword_is_removed_with_a_migration_message():
    with pytest.raises(TypeError, match="RAGSettings"):
        Settings(_env_file=None, rag=RAGSettings(top_k=11))
    assert not hasattr(Settings(_env_file=None), "rag")


# --- Settings must stay generic ---------------------------------------------------

def test_settings_has_no_rag_specific_fields():
    """RAG fields must live only on RAGSettings, never on Settings -- adding a second
    tool to kuhaku must never require touching this class."""

    settings = Settings(_env_file=None)
    rag_only_field_names = {f.name for f in dataclasses.fields(RAGSettings)} - {
        # Cross-tool fields RAGSettings mirrors but that legitimately stay on Settings
        # too (see RAGSettings.from_settings()).
        "vertex_project", "vertex_location", "retry_enabled",
    }
    for name in rag_only_field_names:
        assert not hasattr(settings, name), f"Settings should not carry RAG field {name!r}"
