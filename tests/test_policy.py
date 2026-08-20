"""Tests for the default behavior policy (DECISIONS.md D53): startup security
validation, performance-component fallback, and custom-component fail-fast validation.
"""

from __future__ import annotations

import warnings

import pytest

from kuhaku.core.config import Settings
from kuhaku.core.exceptions import (
    CustomComponentError,
    FallbackWarning,
    SecurityComponentError,
)
from kuhaku.core.policy import (
    apply_fallback_policy,
    enforce_security_policy,
    validate_custom_component,
    validate_startup,
)
from kuhaku.tools.rag.reranker import Reranker
from tests.conftest import FakeGuardPipeline


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- Feature 1: security validation --------------------------------------------------
class _BrokenGuard:
    def validate(self) -> bool:
        raise RuntimeError("stage-1 model file is corrupt")


def test_security_component_error_raised_when_guard_fails_to_initialize(tmp_path):
    settings = _settings(
        guard_enabled=True, audit_log_path=str(tmp_path / "audit.jsonl")
    )

    with pytest.raises(SecurityComponentError, match="GuardPipeline"):
        enforce_security_policy(settings, {"guard": _BrokenGuard()})


def test_security_component_error_raised_when_guard_missing_but_enabled(tmp_path):
    settings = _settings(
        guard_enabled=True, audit_log_path=str(tmp_path / "audit.jsonl")
    )

    with pytest.raises(SecurityComponentError, match="not constructed"):
        enforce_security_policy(settings, {"guard": None})


def test_no_validation_when_guard_explicitly_disabled(tmp_path):
    """Security is explicitly turned off -- no validation should run (D1 edge case)."""

    settings = _settings(
        guard_enabled=False, audit_log_path=str(tmp_path / "audit.jsonl")
    )

    enforce_security_policy(settings, {"guard": None})  # must not raise


def test_working_guard_passes_validation(tmp_path):
    settings = _settings(
        guard_enabled=True, audit_log_path=str(tmp_path / "audit.jsonl")
    )

    enforce_security_policy(settings, {"guard": FakeGuardPipeline()})  # must not raise


def test_security_component_error_when_audit_log_path_not_writable(tmp_path):
    unwritable_parent = tmp_path / "not_a_directory"
    unwritable_parent.write_text("x", encoding="utf-8")  # a file, not a directory

    settings = _settings(
        guard_enabled=False, audit_log_path=str(unwritable_parent / "audit.jsonl")
    )

    with pytest.raises(SecurityComponentError, match="audit log"):
        enforce_security_policy(settings, {})


def test_audit_log_path_empty_uses_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = _settings(audit_log_path="")

    enforce_security_policy(settings, {})
    assert (tmp_path / "logs").is_dir()


def test_audit_log_path_none_uses_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = _settings(audit_log_path=None)

    enforce_security_policy(settings, {})
    assert (tmp_path / "logs").is_dir()


def test_security_component_error_when_default_audit_log_path_not_writable(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").write_text("x", encoding="utf-8")
    settings = _settings(audit_log_path="")

    with pytest.raises(SecurityComponentError, match="audit log"):
        enforce_security_policy(settings, {})


def test_no_audit_validation_when_audit_explicitly_disabled(tmp_path):
    """audit_enabled=False (D53's "explicitly disabled" tier) skips path validation
    entirely -- even an otherwise-unwritable path must not raise."""

    unwritable_parent = tmp_path / "not_a_directory"
    unwritable_parent.write_text("x", encoding="utf-8")  # a file, not a directory
    settings = _settings(
        audit_enabled=False, audit_log_path=str(unwritable_parent / "audit.jsonl")
    )

    enforce_security_policy(settings, {})  # must not raise


def test_no_audit_validation_when_disabled_and_path_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = _settings(audit_enabled=False, audit_log_path=None)

    enforce_security_policy(settings, {})  # must not raise
    assert not (tmp_path / "logs").exists()


def test_audit_enabled_true_still_validates_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = _settings(audit_enabled=True, audit_log_path=None)

    enforce_security_policy(settings, {})
    assert (tmp_path / "logs").is_dir()


def test_validate_startup_returns_components_and_enforces_security(tmp_path):
    settings = _settings(
        guard_enabled=True, audit_log_path=str(tmp_path / "audit.jsonl")
    )
    guard = FakeGuardPipeline()

    result = validate_startup(settings, {"guard": guard})

    assert result == {"guard": guard}


# --- Feature 2: performance component fallback ----------------------------------------
def test_apply_fallback_policy_falls_back_and_warns_when_default_component_fails():
    settings = _settings(strict_performance_components=False)

    def broken_build():
        raise ImportError("rank_bm25 not installed")

    with pytest.warns(FallbackWarning, match="BM25Retriever"):
        result = apply_fallback_policy(
            settings, "BM25Retriever", broken_build, fallback=None
        )

    assert result is None


def test_apply_fallback_policy_returns_build_result_on_success():
    settings = _settings()

    result = apply_fallback_policy(settings, "BM25Retriever", lambda: "sparse", fallback=None)

    assert result == "sparse"


def test_apply_fallback_policy_raises_when_strict_performance_components():
    settings = _settings(strict_performance_components=True)

    def broken_build():
        raise RuntimeError("cross-encoder model download failed")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        with pytest.raises(RuntimeError, match="cross-encoder model download failed"):
            apply_fallback_policy(
                settings, "CrossEncoderReranker", broken_build, fallback=None
            )


# --- Feature 3: custom component validation --------------------------------------------
class _NotAReranker:
    """Missing the required `rerank` method entirely."""


class _ValidCustomReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


def test_custom_component_error_for_invalid_custom_reranker():
    with pytest.raises(CustomComponentError, match="rerank"):
        validate_custom_component(_NotAReranker(), Reranker, component_kind="reranker")


def test_valid_custom_reranker_passes_validation():
    validate_custom_component(
        _ValidCustomReranker(), Reranker, component_kind="reranker"
    )  # must not raise
