"""Tests for kuhaku.core.capabilities: the generic probes and the ``"auto"`` resolver."""

from __future__ import annotations

import warnings

import pytest

from kuhaku.core import capabilities as cap
from kuhaku.core.capabilities import AUTO, Resolution, resolve
from kuhaku.core.config import Settings
from kuhaku.core.exceptions import FallbackWarning
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.anthropic_provider import AnthropicProvider
from kuhaku.core.llm.ollama_provider import OllamaProvider


@pytest.fixture(autouse=True)
def _clear_emitted():
    cap.reset_emitted()
    yield
    cap.reset_emitted()


# --- auto_enabled -------------------------------------------------------------------

def test_auto_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("KUHAKU_AUTO", raising=False)
    assert cap.auto_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
def test_auto_enabled_false_values(monkeypatch, value):
    monkeypatch.setenv("KUHAKU_AUTO", value)
    assert cap.auto_enabled() is False


# --- probes ----------------------------------------------------------------------

def test_module_available():
    assert cap.module_available("json") is True
    assert cap.module_available("nonexistent_module_xyz") is False


def test_endpoint_reachable_false_for_dead_port():
    # 9 is discard; nothing listens on it in CI.
    assert cap.endpoint_reachable("http://127.0.0.1:9", timeout=0.2) is False


# --- resolve -------------------------------------------------------------------------

def test_resolve_returns_pinned_value_untouched():
    assert resolve("x", "pinned", baseline="b", candidates=[("other", lambda: True)]) == "pinned"


def test_resolve_auto_disabled_returns_baseline(monkeypatch):
    monkeypatch.setenv("KUHAKU_AUTO", "false")
    assert resolve("x", AUTO, baseline="b", candidates=[("other", lambda: True)]) == "b"


def test_resolve_picks_first_matching_candidate():
    chosen = resolve(
        "x", AUTO, baseline="b",
        candidates=[("a", lambda: False), ("c", lambda: True), ("d", lambda: True)],
    )
    assert chosen == "c"


def test_resolve_falls_back_to_baseline_when_nothing_matches():
    assert resolve("x", AUTO, baseline="b", candidates=[("a", lambda: False)]) == "b"


def test_resolve_probe_exception_is_treated_as_no_match():
    def boom() -> bool:
        raise RuntimeError("nope")

    assert resolve("x", AUTO, baseline="b", candidates=[("a", boom), ("c", lambda: True)]) == "c"


def test_resolve_emits_notice_on_deviation(capsys):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve("field_y", AUTO, baseline="b", candidates=[("c", lambda: True)])
    assert any(isinstance(w.message, FallbackWarning) for w in caught)
    assert "field_y" in capsys.readouterr().err


def test_resolve_silent_when_baseline_chosen(capsys):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve("field_z", AUTO, baseline="b", candidates=[("c", lambda: False)])
    assert not caught
    assert capsys.readouterr().err == ""


def test_emit_deduplicates(capsys):
    r = Resolution("f", "c", "b", "reason")
    cap.emit(r)
    cap.emit(r)
    assert capsys.readouterr().err.count("[kuhaku]") == 1


# --- llm_provider resolution -------------------------------------------------------

def _settings(**kw) -> Settings:
    base = dict(_env_file=None, llm_provider=AUTO, anthropic_api_key=None, openai_api_key=None)
    base.update(kw)
    return Settings(**base)


def test_llm_auto_prefers_reachable_ollama(monkeypatch):
    monkeypatch.setattr("kuhaku.core.llm.endpoint_reachable", lambda *a, **k: True)
    assert isinstance(build_llm_provider(_settings()), OllamaProvider)


def test_llm_auto_falls_to_provider_with_credentials(monkeypatch):
    monkeypatch.setattr("kuhaku.core.llm.endpoint_reachable", lambda *a, **k: False)
    provider = build_llm_provider(_settings(anthropic_api_key="k"))
    assert isinstance(provider, AnthropicProvider)


def test_llm_auto_disabled_stays_ollama(monkeypatch):
    monkeypatch.setenv("KUHAKU_AUTO", "false")
    monkeypatch.setattr("kuhaku.core.llm.endpoint_reachable", lambda *a, **k: False)
    assert isinstance(build_llm_provider(_settings(anthropic_api_key="k")), OllamaProvider)


def test_llm_explicit_value_wins_over_auto(monkeypatch):
    monkeypatch.setattr("kuhaku.core.llm.endpoint_reachable", lambda *a, **k: False)
    assert isinstance(build_llm_provider(_settings(llm_provider="ollama")), OllamaProvider)
