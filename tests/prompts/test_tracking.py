"""Prompt refs on spans and as metadata."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from kuhaku.core.observability import start_span
from kuhaku.core.prompts import (
    PromptRef,
    prompt_metadata,
    prompt_span_attributes,
    record_prompts,
)

SYSTEM = PromptRef("s", "1", "h1")
USER = PromptRef("u", "2", "h2")


def test_span_attributes_are_keyed_by_role_and_skip_none():
    assert prompt_span_attributes(system=SYSTEM, user=None) == {
        "kuhaku.prompt.system.name": "s",
        "kuhaku.prompt.system.version": "1",
        "kuhaku.prompt.system.hash": "h1",
    }


def test_metadata_shape():
    assert prompt_metadata(system=SYSTEM, user=USER) == {
        "system": {"name": "s", "version": "1", "hash": "h1"},
        "user": {"name": "u", "version": "2", "hash": "h2"},
    }


def test_record_prompts_annotates_the_current_span_and_is_safe_without_one():
    record_prompts(system=SYSTEM)  # no span open: must not raise

    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    exporter.clear()
    with start_span("unit_test_prompt_span"):
        record_prompts(system=SYSTEM, user=USER)

    [span] = [s for s in exporter.get_finished_spans() if s.name == "unit_test_prompt_span"]
    assert span.attributes["kuhaku.prompt.system.version"] == "1"
    assert span.attributes["kuhaku.prompt.user.hash"] == "h2"
