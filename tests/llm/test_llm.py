"""Tests for the LLM provider abstraction: factory, parsing, and error handling.

No real network calls — ``requests.post`` is monkeypatched.
"""

from __future__ import annotations

import pytest
import requests

from kuhaku.core.config import Settings
from kuhaku.core.llm import build_llm_provider
from kuhaku.core.llm.anthropic_provider import AnthropicProvider
from kuhaku.core.llm.base import LLMError, TokenUsage
from kuhaku.core.llm.ollama_provider import OllamaProvider
from kuhaku.core.llm.openai_provider import OpenAIProvider
from kuhaku.core.observability.metrics import (
    RETRY_ATTEMPTS,
    RETRY_FAILURES,
    RETRY_SUCCESSES,
)
from tests.conftest import prometheus_counter_value as _counter_value


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _settings(**overrides) -> Settings:
    base = dict(llm_provider="ollama", anthropic_api_key=None, openai_api_key=None)
    base.update(overrides)
    return Settings(_env_file=None, **base)


# --- factory ----------------------------------------------------------------
def test_factory_builds_ollama():
    provider = build_llm_provider(_settings(llm_provider="ollama"))
    assert isinstance(provider, OllamaProvider)
    assert provider.name.startswith("ollama:")


def test_factory_builds_anthropic():
    provider = build_llm_provider(_settings(llm_provider="anthropic", anthropic_api_key="k"))
    assert isinstance(provider, AnthropicProvider)


def test_factory_builds_openai():
    provider = build_llm_provider(_settings(llm_provider="openai", openai_api_key="k"))
    assert isinstance(provider, OpenAIProvider)


def test_factory_unknown_provider_raises():
    with pytest.raises(LLMError):
        build_llm_provider(_settings(llm_provider="does-not-exist"))


def test_provider_selection_is_case_insensitive():
    assert isinstance(build_llm_provider(_settings(llm_provider="OLLAMA")), OllamaProvider)


# --- default-provider auto-fallback (LLM_PROVIDER left unset) ---------------
def test_default_provider_uses_ollama_when_reachable(monkeypatch):
    monkeypatch.setattr(
        "kuhaku.core.llm._ollama_reachable", lambda base_url, **kwargs: True
    )
    settings = Settings(_env_file=None, anthropic_api_key=None, openai_api_key=None)
    assert "llm_provider" not in settings.model_fields_set
    provider = build_llm_provider(settings)
    assert isinstance(provider, OllamaProvider)


def test_default_provider_falls_back_to_hosted_when_ollama_unreachable(monkeypatch, caplog):
    monkeypatch.setattr(
        "kuhaku.core.llm._ollama_reachable", lambda base_url, **kwargs: False
    )
    settings = Settings(_env_file=None, anthropic_api_key="sk-test", openai_api_key=None)
    with caplog.at_level("WARNING"):
        provider = build_llm_provider(settings)
    assert isinstance(provider, AnthropicProvider)
    assert "falling back to 'anthropic'" in caplog.text


def test_default_provider_raises_clear_error_when_nothing_is_available(monkeypatch):
    monkeypatch.setattr(
        "kuhaku.core.llm._ollama_reachable", lambda base_url, **kwargs: False
    )
    settings = Settings(_env_file=None, anthropic_api_key=None, openai_api_key=None)
    with pytest.raises(LLMError) as exc_info:
        build_llm_provider(settings)
    message = str(exc_info.value)
    assert "KUHAKU_LLM_PROVIDER=anthropic" in message
    assert "ollama pull" in message


def test_explicit_ollama_choice_skips_the_reachability_probe(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("must not probe when LLM_PROVIDER is set explicitly")

    monkeypatch.setattr("kuhaku.core.llm._ollama_reachable", _boom)
    settings = Settings(_env_file=None, llm_provider="ollama", anthropic_api_key=None)
    provider = build_llm_provider(settings)
    assert isinstance(provider, OllamaProvider)


# --- missing credentials ----------------------------------------------------
def test_anthropic_requires_key():
    with pytest.raises(LLMError):
        AnthropicProvider(api_key="", model="m")


def test_openai_requires_key():
    with pytest.raises(LLMError):
        OpenAIProvider(api_key="", model="m")


# --- eager config validation --------------------------------------------------
def test_ollama_requires_base_url():
    with pytest.raises(LLMError):
        OllamaProvider(base_url="", model="m")


def test_ollama_requires_base_url_not_whitespace():
    with pytest.raises(LLMError):
        OllamaProvider(base_url="   ", model="m")


def test_ollama_requires_model():
    with pytest.raises(LLMError):
        OllamaProvider(base_url="http://x", model="")


def test_ollama_requires_model_not_whitespace():
    with pytest.raises(LLMError):
        OllamaProvider(base_url="http://x", model="   ")


def test_openai_requires_model():
    with pytest.raises(LLMError):
        OpenAIProvider(api_key="k", model="")


def test_openai_requires_model_not_whitespace():
    with pytest.raises(LLMError):
        OpenAIProvider(api_key="k", model="   ")


def test_anthropic_requires_model():
    with pytest.raises(LLMError):
        AnthropicProvider(api_key="k", model="")


def test_anthropic_requires_model_not_whitespace():
    with pytest.raises(LLMError):
        AnthropicProvider(api_key="k", model="   ")


# --- generate parsing -------------------------------------------------------
def test_ollama_generate_parses_message(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"message": {"content": " hi "}})
    )
    provider = OllamaProvider("http://x", "m")
    assert provider.generate("sys", "user") == "hi"


def test_anthropic_generate_concatenates_text_blocks(monkeypatch):
    payload = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload))
    provider = AnthropicProvider(api_key="k", model="m")
    assert provider.generate("sys", "user") == "ab"


def test_openai_generate_parses_choice(monkeypatch):
    payload = {"choices": [{"message": {"content": "answer"}}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload))
    provider = OpenAIProvider(api_key="k", model="m")
    assert provider.generate("sys", "user") == "answer"


# --- error handling ---------------------------------------------------------
# retry_enabled=False below: these tests assert the network-error-to-LLMError wrapping,
# not retry behavior (covered separately below) -- disabling retry keeps them instant
# instead of sleeping through real exponential backoff.
def test_network_error_becomes_llmerror(monkeypatch):
    def boom(*a, **k):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(LLMError):
        OllamaProvider("http://x", "m", retry_enabled=False).generate("s", "u")


def test_malformed_response_becomes_llmerror(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({"unexpected": 1}))
    with pytest.raises(LLMError):
        OpenAIProvider(api_key="k", model="m").generate("s", "u")


def test_ollama_malformed_response(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({"nope": 1}))
    with pytest.raises(LLMError):
        OllamaProvider("http://x", "m").generate("s", "u")


def test_anthropic_network_error(monkeypatch):
    def boom(*a, **k):
        raise requests.RequestException("timeout")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(LLMError):
        AnthropicProvider(api_key="k", model="m", retry_enabled=False).generate("s", "u")


def test_anthropic_malformed_response(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({"bad": 1}))
    with pytest.raises(LLMError):
        AnthropicProvider(api_key="k", model="m").generate("s", "u")


def test_openai_network_error(monkeypatch):
    def boom(*a, **k):
        raise requests.RequestException("dns")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(LLMError):
        OpenAIProvider(api_key="k", model="m", retry_enabled=False).generate("s", "u")


def test_provider_names():
    assert OllamaProvider("http://x", "m").name == "ollama:m"
    assert AnthropicProvider(api_key="k", model="m").name == "anthropic:m"
    assert OpenAIProvider(api_key="k", model="m").name == "openai:m"


# --- best-effort token usage logging (never changes the return value) --------
def test_ollama_logs_token_usage_when_present(monkeypatch, caplog):
    import logging

    payload = {"message": {"content": "hi"}, "eval_count": 12, "prompt_eval_count": 34}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload))
    with caplog.at_level(logging.INFO):
        result = OllamaProvider("http://x", "m").generate("s", "u")
    assert result == "hi"  # return type/value unaffected
    usage_records = [r for r in caplog.records if getattr(r, "token_count", None) == 12]
    assert usage_records and usage_records[0].prompt_token_count == 34


def test_ollama_no_usage_fields_no_log(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"message": {"content": "hi"}})
    )
    with caplog.at_level(logging.INFO):
        OllamaProvider("http://x", "m").generate("s", "u")
    assert not any(hasattr(r, "token_count") for r in caplog.records)


def test_anthropic_logs_token_usage_when_present(monkeypatch, caplog):
    import logging

    payload = {
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload))
    with caplog.at_level(logging.INFO):
        result = AnthropicProvider(api_key="k", model="m").generate("s", "u")
    assert result == "hi"
    usage_records = [r for r in caplog.records if getattr(r, "token_count", None) == 5]
    assert usage_records and usage_records[0].prompt_token_count == 10


def test_openai_logs_token_usage_when_present(monkeypatch, caplog):
    import logging

    payload = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload))
    with caplog.at_level(logging.INFO):
        result = OpenAIProvider(api_key="k", model="m").generate("s", "u")
    assert result == "hi"
    usage_records = [r for r in caplog.records if getattr(r, "token_count", None) == 3]
    assert usage_records and usage_records[0].prompt_token_count == 7


# --- FR3: last_usage side channel for token tracking --------------------------
def test_ollama_sets_last_usage(monkeypatch):
    payload = {"message": {"content": "hi"}, "eval_count": 12, "prompt_eval_count": 34}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload))
    provider = OllamaProvider("http://x", "m")
    assert provider.last_usage is None  # nothing recorded before the first call
    provider.generate("s", "u")
    assert provider.last_usage == TokenUsage(prompt_tokens=34, completion_tokens=12)


def test_ollama_last_usage_stays_none_without_usage_fields(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"message": {"content": "hi"}})
    )
    provider = OllamaProvider("http://x", "m")
    provider.generate("s", "u")
    assert provider.last_usage is None


# --- Retry (D40) --------------------------------------------------------------
class FailingHTTPResponse:
    """Raises requests.HTTPError from raise_for_status(), with a .response.status_code
    a retry predicate can inspect -- mirrors what a real `requests.Response` exposes."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self):
        err = requests.HTTPError(f"{self.status_code} error")
        err.response = self  # only `.status_code` is read by the retry predicate
        raise err

    def json(self):  # pragma: no cover - never reached, raise_for_status always raises
        return {}


def _flaky_post(fail_times: int, success_payload: dict):
    """A `requests.post` stand-in that fails with a connection error `fail_times`
    times, then returns `success_payload`. `calls` lets tests assert the attempt count."""

    calls: list[int] = []

    def _post(*a, **k):
        calls.append(1)
        if len(calls) <= fail_times:
            raise requests.ConnectionError(f"transient failure #{len(calls)}")
        return FakeResponse(success_payload)

    return _post, calls


_FAST_RETRY = dict(
    retry_backoff_base_seconds=0.001,
    retry_backoff_max_seconds=0.01,
)


def test_ollama_retries_transient_failure_then_succeeds(monkeypatch):
    post, calls = _flaky_post(2, {"message": {"content": "hi"}})
    monkeypatch.setattr(requests, "post", post)

    provider = OllamaProvider("http://x", "m", retry_max_attempts=3, **_FAST_RETRY)
    before_attempts = _counter_value(RETRY_ATTEMPTS, service="llm")
    before_successes = _counter_value(RETRY_SUCCESSES, service="llm")

    assert provider.generate("s", "u") == "hi"
    assert len(calls) == 3
    assert _counter_value(RETRY_ATTEMPTS, service="llm") == before_attempts + 2
    assert _counter_value(RETRY_SUCCESSES, service="llm") == before_successes + 1


def test_ollama_exhausts_all_retries_then_raises_llmerror(monkeypatch):
    post, calls = _flaky_post(99, {"message": {"content": "hi"}})  # never succeeds
    monkeypatch.setattr(requests, "post", post)

    provider = OllamaProvider("http://x", "m", retry_max_attempts=3, **_FAST_RETRY)
    before_failures = _counter_value(RETRY_FAILURES, service="llm")

    with pytest.raises(LLMError):
        provider.generate("s", "u")
    assert len(calls) == 3
    assert _counter_value(RETRY_FAILURES, service="llm") == before_failures + 1


def test_ollama_4xx_is_not_retried(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FailingHTTPResponse(400))
    provider = OllamaProvider("http://x", "m", retry_max_attempts=3, **_FAST_RETRY)

    before_attempts = _counter_value(RETRY_ATTEMPTS, service="llm")
    with pytest.raises(LLMError):
        provider.generate("s", "u")
    # No retry recorded: a 4xx short-circuits after exactly one attempt.
    assert _counter_value(RETRY_ATTEMPTS, service="llm") == before_attempts


def test_ollama_5xx_is_retried(monkeypatch):
    calls: list[int] = []

    def _post(*a, **k):
        calls.append(1)
        return FailingHTTPResponse(503) if len(calls) == 1 else FakeResponse(
            {"message": {"content": "hi"}}
        )

    monkeypatch.setattr(requests, "post", _post)
    provider = OllamaProvider("http://x", "m", retry_max_attempts=3, **_FAST_RETRY)
    assert provider.generate("s", "u") == "hi"
    assert len(calls) == 2


def test_retry_disabled_setting_bypasses_retry_entirely(monkeypatch):
    post, calls = _flaky_post(2, {"message": {"content": "hi"}})
    monkeypatch.setattr(requests, "post", post)

    provider = OllamaProvider("http://x", "m", retry_enabled=False, **_FAST_RETRY)
    with pytest.raises(LLMError):
        provider.generate("s", "u")
    assert len(calls) == 1


def test_anthropic_retries_transient_failure_then_succeeds(monkeypatch):
    """Confirms retry applies to Anthropic too, not just Ollama (D40 resolution)."""

    payload = {"content": [{"type": "text", "text": "hi"}]}
    post, calls = _flaky_post(1, payload)
    monkeypatch.setattr(requests, "post", post)

    provider = AnthropicProvider(
        api_key="k", model="m", retry_max_attempts=3, **_FAST_RETRY
    )
    assert provider.generate("s", "u") == "hi"
    assert len(calls) == 2


def test_openai_retries_transient_failure_then_succeeds(monkeypatch):
    """Confirms retry applies to OpenAI too, not just Ollama (D40 resolution)."""

    payload = {"choices": [{"message": {"content": "hi"}}]}
    post, calls = _flaky_post(1, payload)
    monkeypatch.setattr(requests, "post", post)

    provider = OpenAIProvider(
        api_key="k", model="m", retry_max_attempts=3, **_FAST_RETRY
    )
    assert provider.generate("s", "u") == "hi"
    assert len(calls) == 2


def test_openai_sets_last_usage(monkeypatch):
    payload = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload))
    provider = OpenAIProvider(api_key="k", model="m")
    provider.generate("s", "u")
    assert provider.last_usage == TokenUsage(prompt_tokens=7, completion_tokens=3)


def test_anthropic_sets_last_usage(monkeypatch):
    payload = {
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload))
    provider = AnthropicProvider(api_key="k", model="m")
    provider.generate("s", "u")
    assert provider.last_usage == TokenUsage(prompt_tokens=10, completion_tokens=5)


# --- Circuit breaker -----------------------------------------------------------
def test_ollama_circuit_opens_after_threshold_and_rejects_without_calling(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    provider = OllamaProvider(
        "http://x",
        "m",
        retry_enabled=False,
        circuit_breaker_failure_threshold=2,
        circuit_breaker_reset_timeout_seconds=60.0,
    )

    for _ in range(2):
        with pytest.raises(LLMError):
            provider.generate("s", "u")

    calls = {"n": 0}
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    with pytest.raises(LLMError):
        provider.generate("s", "u")
    assert calls["n"] == 0  # circuit open -- the HTTP call was never attempted


def test_circuit_breaker_disabled_never_short_circuits(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    provider = OllamaProvider(
        "http://x",
        "m",
        retry_enabled=False,
        circuit_breaker_enabled=False,
    )

    for _ in range(10):
        with pytest.raises(LLMError):
            provider.generate("s", "u")

    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"message": {"content": "hi"}})
    )
    assert provider.generate("s", "u") == "hi"  # never opened -- call still reaches requests.post


def test_openai_circuit_opens_after_threshold(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    provider = OpenAIProvider(
        api_key="k",
        model="m",
        retry_enabled=False,
        circuit_breaker_failure_threshold=1,
        circuit_breaker_reset_timeout_seconds=60.0,
    )

    with pytest.raises(LLMError):
        provider.generate("s", "u")

    calls = {"n": 0}
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    with pytest.raises(LLMError):
        provider.generate("s", "u")
    assert calls["n"] == 0


def test_anthropic_circuit_opens_after_threshold(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    provider = AnthropicProvider(
        api_key="k",
        model="m",
        retry_enabled=False,
        circuit_breaker_failure_threshold=1,
        circuit_breaker_reset_timeout_seconds=60.0,
    )

    with pytest.raises(LLMError):
        provider.generate("s", "u")

    calls = {"n": 0}
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    with pytest.raises(LLMError):
        provider.generate("s", "u")
    assert calls["n"] == 0


def test_circuit_recovers_after_reset_timeout(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    provider = OllamaProvider(
        "http://x",
        "m",
        retry_enabled=False,
        circuit_breaker_failure_threshold=1,
        circuit_breaker_reset_timeout_seconds=0.01,
    )

    with pytest.raises(LLMError):
        provider.generate("s", "u")

    import time

    time.sleep(0.3)
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"message": {"content": "hi"}})
    )
    assert provider.generate("s", "u") == "hi"  # half-open probe succeeds -- circuit closes


def test_factory_wires_resilience_settings_through_to_provider():
    settings = _settings(
        llm_provider="ollama",
        llm_timeout_seconds=7,
        circuit_breaker_enabled=False,
    )
    provider = build_llm_provider(settings)
    assert provider._timeout == 7  # noqa: SLF001
    assert provider._circuit_breaker is None  # noqa: SLF001
