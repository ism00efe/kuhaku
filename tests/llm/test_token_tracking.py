"""Tests for the TokenTrackingLLM decorator (FR3)."""

from __future__ import annotations

import logging

from kuhaku.core.llm.base import TokenUsage
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.observability.metrics import LLM_TOKENS_TOTAL
from kuhaku.core.observability.tracing import GEN_AI_SYSTEM, GEN_AI_TOKEN_TYPE
from tests.conftest import prometheus_counter_value as _counter_value


class _FakeProvider:
    """LLMProvider stub whose `last_usage` a test can set directly."""

    def __init__(self, response: str = "answer", last_usage: TokenUsage | None = None) -> None:
        self.response = response
        self.last_usage = last_usage
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "fake:model-x"

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


def test_generate_delegates_and_returns_unchanged():
    wrapped = _FakeProvider(response="hello [S1]")
    tracker = TokenTrackingLLM(wrapped, provider="openai")
    assert tracker.generate("sys", "usr") == "hello [S1]"
    assert wrapped.calls == [("sys", "usr")]


def test_name_delegates_to_wrapped():
    tracker = TokenTrackingLLM(_FakeProvider(), provider="openai")
    assert tracker.name == "fake:model-x"


def test_records_tokens_when_usage_present():
    wrapped = _FakeProvider(last_usage=TokenUsage(prompt_tokens=100, completion_tokens=50))
    tracker = TokenTrackingLLM(wrapped, provider="openai")

    before_input = _counter_value(
        LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "openai", GEN_AI_TOKEN_TYPE: "input"}
    )
    before_output = _counter_value(
        LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "openai", GEN_AI_TOKEN_TYPE: "output"}
    )

    tracker.generate("sys", "usr")

    assert (
        _counter_value(LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "openai", GEN_AI_TOKEN_TYPE: "input"})
        == before_input + 100
    )
    assert (
        _counter_value(LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "openai", GEN_AI_TOKEN_TYPE: "output"})
        == before_output + 50
    )


def test_ollama_tokens_still_counted():
    wrapped = _FakeProvider(last_usage=TokenUsage(prompt_tokens=10, completion_tokens=5))
    tracker = TokenTrackingLLM(wrapped, provider="ollama")

    before = _counter_value(
        LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "ollama", GEN_AI_TOKEN_TYPE: "input"}
    )
    tracker.generate("sys", "usr")
    assert (
        _counter_value(LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "ollama", GEN_AI_TOKEN_TYPE: "input"})
        == before + 10
    )


def test_no_usage_reported_records_nothing():
    """A provider that never set `last_usage` (e.g. a plain test fake) is simply not
    token-tracked -- must not raise or fabricate usage."""

    wrapped = _FakeProvider(last_usage=None)
    tracker = TokenTrackingLLM(wrapped, provider="openai")

    before = _counter_value(
        LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "openai", GEN_AI_TOKEN_TYPE: "input"}
    )
    result = tracker.generate("sys", "usr")
    assert result == "answer"
    assert (
        _counter_value(LLM_TOKENS_TOTAL, **{GEN_AI_SYSTEM: "openai", GEN_AI_TOKEN_TYPE: "input"})
        == before
    )


def test_provider_without_last_usage_attribute_is_not_token_tracked():
    """Any LLMProvider (not just the three built-in ones) can be wrapped -- one that
    never sets `last_usage` at all (not even to None) must not raise."""

    class _MinimalProvider:
        @property
        def name(self) -> str:
            return "minimal"

        def generate(self, system: str, user: str) -> str:
            return "ok"

    tracker = TokenTrackingLLM(_MinimalProvider(), provider="openai")
    assert tracker.generate("s", "u") == "ok"


def test_token_recording_failure_is_swallowed_not_propagated(monkeypatch, caplog):
    """FR3: token tracking must be non-blocking -- a metrics failure must not turn a
    successful answer into a failed request."""

    wrapped = _FakeProvider(last_usage=TokenUsage(prompt_tokens=1, completion_tokens=1))
    tracker = TokenTrackingLLM(wrapped, provider="openai")

    def boom(*args, **kwargs):
        raise RuntimeError("metrics backend down")

    monkeypatch.setattr(LLM_TOKENS_TOTAL, "add", boom)

    with caplog.at_level(logging.ERROR):
        result = tracker.generate("sys", "usr")

    assert result == "answer"  # the real answer still comes back
    assert any("token tracking failed" in r.message for r in caplog.records)


def test_logs_usage_at_info_level(caplog):
    wrapped = _FakeProvider(last_usage=TokenUsage(prompt_tokens=100, completion_tokens=50))
    tracker = TokenTrackingLLM(wrapped, provider="openai")

    with caplog.at_level(logging.INFO):
        tracker.generate("sys", "usr")

    records = [r for r in caplog.records if getattr(r, "prompt_token_count", None) is not None]
    assert records
    assert records[0].provider == "openai"
    assert records[0].model == "fake:model-x"
    assert records[0].prompt_token_count == 100
    assert records[0].token_count == 50
