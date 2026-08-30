"""``build_llm_provider`` through the capability resolver: the ``auto`` chain, pinning,
and the non-interactive multi-candidate path.

The generic resolver mechanism is covered under tests/resolve/; this file is the
LLM-wiring counterpart of the old kuhaku.core.capabilities llm tests.
"""

from __future__ import annotations

import pytest

from kuhaku.core.config import Settings
from kuhaku.core.exceptions import CapabilityUnavailable
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.anthropic_provider import AnthropicProvider
from kuhaku.core.llm.groq_provider import GroqProvider
from kuhaku.core.llm.ollama_provider import OllamaProvider
from kuhaku.core.resolve.adapters import llm as llm_adapters
from tests.resolve.conftest import FakeMemory, FakeUI


def _settings(**kw) -> Settings:
    base = dict(
        _env_file=None, llm_provider="auto",
        anthropic_api_key=None, openai_api_key=None, groq_api_key=None,
    )
    base.update(kw)
    return Settings(**base)


def _build(settings, *, reachable, ui=None):
    monkey_ui = ui or FakeUI(interactive=False)
    return build_llm_provider(
        settings,
        ui=monkey_ui,
        memory=FakeMemory(),
    ), monkey_ui


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    monkeypatch.setattr(llm_adapters, "_ollama_reachable", lambda url: False)


def test_auto_prefers_reachable_ollama(monkeypatch):
    monkeypatch.setattr(llm_adapters, "_ollama_reachable", lambda url: True)
    provider, _ = _build(_settings(), reachable=True)
    assert isinstance(provider, OllamaProvider)


def test_auto_falls_to_the_one_credentialed_provider():
    provider, ui = _build(_settings(anthropic_api_key="k"), reachable=False)
    assert isinstance(provider, AnthropicProvider)  # only one ready -> only_option


def test_auto_noninteractive_many_candidates_picks_safest_and_flags():
    # ollama unreachable; anthropic + groq both credentialed -> groq has the lower
    # safety_rank among ready ones? no -- historical order keeps groq last, so anthropic
    # (rank 11) beats groq (rank 13). The skipped decision is announced prominently.
    provider, ui = _build(
        _settings(anthropic_api_key="k", groq_api_key="g"), reachable=False
    )
    assert isinstance(provider, AnthropicProvider)
    assert any(prominent for _m, prominent, _d in ui.announcements)


def test_pinned_value_wins_over_auto():
    provider, _ = _build(_settings(llm_provider="ollama"), reachable=False)
    assert isinstance(provider, OllamaProvider)


def test_pinned_groq_builds_groq():
    provider, _ = _build(_settings(llm_provider="groq", groq_api_key="g"), reachable=False)
    assert isinstance(provider, GroqProvider)


def test_auto_disabled_uses_ollama_baseline(monkeypatch):
    monkeypatch.setenv("KUHAKU_AUTO", "false")
    provider, _ = _build(_settings(anthropic_api_key="k"), reachable=False)
    assert isinstance(provider, OllamaProvider)  # documented baseline, no probing


def test_auto_nothing_available_raises_capability_unavailable():
    with pytest.raises(CapabilityUnavailable):
        _build(_settings(), reachable=False)
