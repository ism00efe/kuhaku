"""``auto``: whatever key is set is used, the provider used last wins and is announced,
choosing an LLM never probes the GPU; plus OpenRouter and with_runtime."""

from __future__ import annotations

import pytest

from kuhaku.core.config import Settings
from kuhaku.core.llm import build_llm_provider, with_runtime
from kuhaku.core.llm.base import LLMError
from kuhaku.core.llm.groq_provider import GroqProvider
from kuhaku.core.llm.openai_provider import OpenAIProvider
from kuhaku.core.llm.openrouter_provider import OpenRouterProvider
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.resolve import probes
from kuhaku.core.resolve.adapters import llm as llm_adapters
from tests.resolve.conftest import FakeMemory, FakeUI

NO_KEYS = {
    "openai_api_key": None,
    "anthropic_api_key": None,
    "groq_api_key": None,
    "openrouter_api_key": None,
    "vertex_project": None,
}


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **{**NO_KEYS, **kw})


def _forbidden(*_a, **_k):
    raise AssertionError("choosing an LLM must not probe the GPU")


@pytest.fixture(autouse=True)
def _no_ollama_no_gpu(monkeypatch):
    monkeypatch.setattr(llm_adapters, "_ollama_reachable", lambda url: False)
    monkeypatch.setattr(probes, "gpu_kind", _forbidden)
    monkeypatch.setattr("kuhaku.core.resolve.environment.gpu_kind", _forbidden)


def _build(settings, memory=None, ui=None):
    return build_llm_provider(
        settings, ui=ui or FakeUI(), memory=memory if memory is not None else FakeMemory()
    )


def test_the_default_is_auto():
    assert _settings().llm_provider == "auto"


def test_auto_uses_the_one_key_that_is_set():
    assert isinstance(_build(_settings(openai_api_key="k")), OpenAIProvider)


def test_auto_prefers_the_free_openrouter_over_a_paid_provider():
    provider = _build(_settings(openai_api_key="k", openrouter_api_key="r"))
    assert isinstance(provider, OpenRouterProvider)
    assert provider.name.endswith(":free")


def test_the_provider_used_last_wins_and_is_announced():
    memory, ui = FakeMemory(), FakeUI()
    _build(_settings(llm_provider="groq", groq_api_key="g"), memory=memory)

    provider = _build(_settings(groq_api_key="g", openrouter_api_key="r"), memory=memory, ui=ui)

    assert isinstance(provider, GroqProvider)
    assert any("last one used" in message for message, _p, _d in ui.announcements)


def test_a_pinned_provider_probes_nothing_else(monkeypatch):
    monkeypatch.setattr(llm_adapters, "_ollama_reachable", _forbidden)
    assert isinstance(_build(_settings(llm_provider="openai", openai_api_key="k")), OpenAIProvider)


def test_pinned_openrouter_without_a_key_names_the_key():
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        _build(_settings(llm_provider="openrouter"))


def test_an_unknown_pinned_provider_lists_the_real_ones():
    with pytest.raises(LLMError, match="openrouter"):
        _build(_settings(llm_provider="nope"))


class _Named:
    name = "myprovider:my-model"

    def generate(self, system: str, user: str) -> str:
        return ""


def test_with_runtime_adds_token_tracking_once():
    wrapped = with_runtime(_Named())

    assert isinstance(wrapped, TokenTrackingLLM)
    assert with_runtime(wrapped) is wrapped


def test_with_runtime_can_leave_token_tracking_out():
    provider = _Named()
    assert with_runtime(provider, token_tracking=False) is provider


def test_openai_compatible_providers_name_their_own_circuit_breaker():
    groq = GroqProvider("k", "m")
    openrouter = OpenRouterProvider("k", "m")
    assert groq._circuit_breaker._name == "llm:groq"
    assert openrouter._circuit_breaker._name == "llm:openrouter"
