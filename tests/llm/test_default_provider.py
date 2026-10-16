"""``auto``: whatever key is set is used, the provider used last wins and is announced,
choosing an LLM never probes the GPU; plus OpenRouter and with_runtime."""

from __future__ import annotations

import pytest

from kuhaku.core.config import Settings
from kuhaku.core.exceptions import CapabilityUnavailable
from kuhaku.core.llm import build_llm_provider, with_runtime
from kuhaku.core.llm.base import LLMError
from kuhaku.core.llm.groq_provider import GroqProvider
from kuhaku.core.llm.ollama_provider import OllamaProvider
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



# --- a local Ollama: after every key, and only with a model the user pulled -----------


def _ollama_with(monkeypatch, *models):
    monkeypatch.setattr(llm_adapters, "_ollama_reachable", lambda url: True)
    monkeypatch.setattr(llm_adapters, "_ollama_models", lambda url: list(models))


def test_a_set_key_wins_over_a_running_ollama(monkeypatch):
    _ollama_with(monkeypatch, "llama3.2:latest")
    ui = FakeUI(interactive=True, ask_returns="ollama:llama3.2:latest")

    provider = _build(_settings(openrouter_api_key="r"), ui=ui)

    assert isinstance(provider, OpenRouterProvider)
    assert ui.ask_calls == []  # nothing to ask: the key decides


def test_without_a_key_ollama_runs_the_one_pulled_model(monkeypatch):
    _ollama_with(monkeypatch, "llama3.2:latest")
    provider = _build(_settings())
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama:llama3.2:latest"


def test_several_pulled_models_are_asked_about_in_a_terminal(monkeypatch):
    _ollama_with(monkeypatch, "llama3.2:latest", "gemma3:4b")
    ui = FakeUI(interactive=True, ask_returns="ollama:gemma3:4b")
    memory = FakeMemory()

    assert _build(_settings(), memory=memory, ui=ui).name == "ollama:gemma3:4b"
    assert [o.id for o in ui.ask_calls[0][1]] == ["ollama:llama3.2:latest", "ollama:gemma3:4b"]
    # The answer is remembered: the next run does not ask again.
    again = FakeUI(interactive=True)
    assert _build(_settings(), memory=memory, ui=again).name == "ollama:gemma3:4b"
    assert again.ask_calls == []


def test_without_a_terminal_the_most_recently_pulled_model_is_used_and_announced(monkeypatch):
    _ollama_with(monkeypatch, "llama3.2:latest", "gemma3:4b")
    ui = FakeUI(interactive=False)
    assert _build(_settings(), ui=ui).name == "ollama:llama3.2:latest"
    assert any(prominent for _m, prominent, _d in ui.announcements)


def test_ollama_used_last_keeps_winning_over_a_key(monkeypatch):
    _ollama_with(monkeypatch, "llama3.2:latest")
    memory = FakeMemory()
    _build(_settings(), memory=memory)

    assert isinstance(_build(_settings(openrouter_api_key="r"), memory=memory), OllamaProvider)


def test_a_server_with_no_model_is_not_usable(monkeypatch):
    _ollama_with(monkeypatch)
    with pytest.raises(CapabilityUnavailable, match="no chat model pulled"):
        _build(_settings())


def test_a_configured_model_that_is_not_pulled_is_not_usable(monkeypatch):
    _ollama_with(monkeypatch, "llama3.2:latest")
    with pytest.raises(CapabilityUnavailable, match="qwen3 is not pulled"):
        _build(_settings(ollama_model="qwen3"))


def test_pinned_ollama_without_a_model_uses_a_pulled_one(monkeypatch):
    _ollama_with(monkeypatch, "llama3.2:latest")
    assert _build(_settings(llm_provider="ollama")).name == "ollama:llama3.2:latest"


def test_pinned_ollama_with_a_model_probes_nothing(monkeypatch):
    monkeypatch.setattr(llm_adapters, "_ollama_reachable", _forbidden)
    assert _build(_settings(llm_provider="ollama", ollama_model="m")).name == "ollama:m"


class _Tags:
    def __init__(self, *entries):
        self.entries = entries

    def raise_for_status(self):
        pass

    def json(self):
        return {"models": list(self.entries)}


def test_pulled_models_are_listed_newest_first_without_embedding_models(monkeypatch):
    import requests

    tags = _Tags(
        {"name": "old:1b", "modified_at": "2025-01-01T00:00:00Z", "details": {"family": "llama"}},
        {"name": "nomic-embed-text:latest", "modified_at": "2026-09-01T00:00:00Z",
         "details": {"family": "nomic-bert"}},
        {"name": "new:4b", "modified_at": "2026-10-01T00:00:00Z", "details": {"family": "gemma3"}},
    )
    monkeypatch.setattr(requests, "get", lambda url, timeout: tags)
    assert llm_adapters._ollama_models("http://localhost:11434") == ["new:4b", "old:1b"]


def test_an_unanswering_server_lists_no_models(monkeypatch):
    import requests

    def refuse(url, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", refuse)
    assert llm_adapters._ollama_models("http://localhost:11434") == []
