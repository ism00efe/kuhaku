"""Tests for OTel span creation (D57): `instrumented_step`, `start_span`, and
`TokenTrackingLLM`'s GenAI attribute annotation.

Uses the standard OTel test pattern: attach a fresh ``InMemorySpanExporter`` to the
already-installed global ``TracerProvider`` (``telemetry.py`` always installs one, with no
span processor by default -- see that module's docstring) via ``add_span_processor``, clear
it right after attaching, then exercise the code under test and inspect
``exporter.get_finished_spans()``. Span processors accumulate for the lifetime of the test
process (removal isn't part of the OTel API), so each test uses its own exporter instance
and only asserts on spans it can identify by name/attributes -- never a bare span count,
since unrelated spans from elsewhere in the same process may also reach an
already-attached exporter from an earlier test.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from kuhaku.core.llm.base import TokenUsage
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.observability import instrumented_step
from kuhaku.core.observability.tracing import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    start_span,
)


def _attached_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    exporter.clear()
    return exporter


def test_default_provider_creates_spans_without_a_configured_exporter():
    """Zero-config default: spans are created (real, recording, attribute-bearing) even
    with no span processor attached -- this must never raise or block."""

    with start_span("unit_test_default_span", foo="bar") as span:
        assert span.is_recording()


def test_instrumented_step_creates_a_span_named_after_the_stage_with_attributes():
    exporter = _attached_exporter()

    with instrumented_step("unit_test_traced_stage", initial=1) as rec:
        rec.set(chunk_count=3)

    spans = [s for s in exporter.get_finished_spans() if s.name == "unit_test_traced_stage"]
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["initial"] == 1
    assert span.attributes["chunk_count"] == 3
    assert span.status.status_code == StatusCode.UNSET


def test_instrumented_step_span_gets_error_status_on_exception():
    exporter = _attached_exporter()

    try:
        with instrumented_step("unit_test_failing_stage"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    spans = [s for s in exporter.get_finished_spans() if s.name == "unit_test_failing_stage"]
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


def test_nested_instrumented_steps_are_parent_and_child():
    exporter = _attached_exporter()

    with instrumented_step("unit_test_outer"):
        with instrumented_step("unit_test_inner"):
            pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    outer = spans["unit_test_outer"]
    inner = spans["unit_test_inner"]
    assert inner.parent.span_id == outer.context.span_id


class _FakeProvider:
    """Minimal LLMProvider stub carrying a preset `last_usage`, like conftest's but
    local to this test to keep the tracing suite self-contained."""

    def __init__(self, last_usage: TokenUsage | None) -> None:
        self.last_usage = last_usage

    @property
    def name(self) -> str:
        return "ollama:qwen2.5:7b-instruct"

    def generate(self, system: str, user: str) -> str:
        return "answer"


def test_token_tracking_llm_annotates_the_ambient_span_with_gen_ai_attributes():
    exporter = _attached_exporter()
    wrapped = _FakeProvider(TokenUsage(prompt_tokens=12, completion_tokens=34))
    tracker = TokenTrackingLLM(wrapped, provider="ollama")

    with start_span("unit_test_generate"):
        tracker.generate("sys", "usr")

    spans = [s for s in exporter.get_finished_spans() if s.name == "unit_test_generate"]
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[GEN_AI_SYSTEM] == "ollama"
    assert attrs[GEN_AI_REQUEST_MODEL] == "qwen2.5:7b-instruct"
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 34


def test_token_tracking_llm_is_a_noop_with_no_ambient_span():
    """No span open -- `set_current_span_attributes` must not raise (Feature 3's edge
    case: a tracing failure must never break the main request)."""

    wrapped = _FakeProvider(TokenUsage(prompt_tokens=1, completion_tokens=1))
    tracker = TokenTrackingLLM(wrapped, provider="ollama")
    assert tracker.generate("sys", "usr") == "answer"
