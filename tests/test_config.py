"""Tests for LLM resilience settings and load_settings."""

from __future__ import annotations

from kuhaku.core.config import Settings, load_settings


def test_llm_resilience_defaults():
    settings = Settings(_env_file=None)
    assert settings.llm_timeout_seconds == 120
    assert settings.circuit_breaker_enabled is True
    assert settings.circuit_breaker_failure_threshold == 5
    assert settings.circuit_breaker_reset_timeout_seconds == 60.0
    assert settings.circuit_breaker_success_threshold == 1


def test_llm_resilience_settings_are_overridable():
    settings = Settings(
        _env_file=None,
        llm_timeout_seconds=30,
        circuit_breaker_enabled=False,
        circuit_breaker_failure_threshold=10,
        circuit_breaker_reset_timeout_seconds=15.0,
        circuit_breaker_success_threshold=3,
    )
    assert settings.llm_timeout_seconds == 30
    assert settings.circuit_breaker_enabled is False
    assert settings.circuit_breaker_failure_threshold == 10
    assert settings.circuit_breaker_reset_timeout_seconds == 15.0
    assert settings.circuit_breaker_success_threshold == 3


def test_settings_has_no_application_specific_fields():
    """Settings is kuhaku-level, cross-tool configuration. HTTP-API and app-layer
    fields (host/port/UI language, SSL, per-route rate limits, admin/upload limits)
    belong to the application that embeds kuhaku, not to kuhaku itself --
    see the module docstring on `Settings`."""

    removed_fields = {
        "app_host",
        "app_port",
        "ui_language",
        "app_ssl_keyfile",
        "app_ssl_certfile",
        "api_max_body_bytes",
        "api_rate_limit_enabled",
        "api_rate_limit_requests",
        "api_rate_limit_window_seconds",
        "api_trusted_proxy_hops",
        "api_max_upload_bytes",
        "api_upload_rate_limit_requests",
        "api_upload_rate_limit_window_seconds",
        "api_admin_create_user_rate_limit_requests",
        "api_admin_create_user_rate_limit_window_seconds",
        "api_admin_reset_password_rate_limit_requests",
        "api_admin_reset_password_rate_limit_window_seconds",
        "api_admin_ingest_rate_limit_requests",
        "api_admin_ingest_rate_limit_window_seconds",
        "api_admin_eval_run_rate_limit_requests",
        "api_admin_eval_run_rate_limit_window_seconds",
        "api_admin_upload_max_bytes",
        "login_rate_limit_requests",
        "login_rate_limit_window_seconds",
    }
    present = removed_fields & Settings.model_fields.keys()
    assert not present, f"application-specific fields leaked back into Settings: {present}"


# --- load_settings (D46) -----------------------------------------------------------
def test_load_settings_without_loader_returns_defaults():
    settings = load_settings(env_file=None)
    assert isinstance(settings, Settings)
    # "auto" is the shipped default now; kuhaku.core.capabilities resolves it to a
    # concrete provider at build time (see tests/test_capabilities.py).
    assert settings.llm_provider == "auto"


def test_load_settings_app_config_loader_overrides_settings():
    loader_called = False

    def fake_loader() -> dict[str, str]:
        nonlocal loader_called
        loader_called = True
        return {
            "llm_provider": "openai",
            "llm_temperature": "0.8",
            "llm_max_tokens": "2048",
            "strict_performance_components": "true",
        }

    settings = load_settings(app_config_loader=fake_loader, env_file=None)
    assert loader_called is True
    assert settings.llm_provider == "openai"
    assert settings.llm_temperature == 0.8
    assert settings.llm_max_tokens == 2048
    assert settings.strict_performance_components is True


def test_load_settings_app_config_loader_ignores_bootstrap_and_unknown_keys():
    def fake_loader() -> dict[str, str]:
        return {
            "anthropic_api_key": "injected-key",
            "openai_api_key": "injected-key",
            "ollama_base_url": "http://injected-url:11434",
            "unknown_nonexistent_field": "some_value",
            "llm_temperature": "0.5",
        }

    settings = load_settings(app_config_loader=fake_loader, env_file=None)
    # Bootstrap keys must NOT be overridden by app_config
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    # Non-bootstrap valid field IS overridden
    assert settings.llm_temperature == 0.5


def test_load_settings_empty_app_config_loader_returns_base():
    settings = load_settings(app_config_loader=lambda: {}, env_file=None)
    assert isinstance(settings, Settings)
    assert settings.llm_temperature == 0.1

