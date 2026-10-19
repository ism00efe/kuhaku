"""The RAG system prompt through the prompt registry: overrides, refs, spans, audit."""

from __future__ import annotations

import json

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from kuhaku.core.config import Settings
from kuhaku.core.prompts import (
    CUSTOM_VERSION,
    PromptRef,
    PromptRegistry,
    PromptRenderError,
    PromptTemplate,
)
from kuhaku.core.security.guard import CANARY_TOKEN
from kuhaku.tools.rag.engine import RAGEngine
from kuhaku.tools.rag.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NAME,
    SYSTEM_PROMPT_REF,
    SYSTEM_PROMPT_TEMPLATE_VERSION,
    builtin_prompts,
    render_system_prompt,
)
from tests.conftest import (
    FakeEmbeddings,
    FakeLLM,
    FakeRetriever,
    FakeVectorStore,
    make_chunk,
)

_MARKERS = (
    "{{PERSONA}}\n{{LANGUAGE_POLICY}}\n{{FORMAT_PREFERENCE}}\n{{EXAMPLE}}\n"
    "Canary {{CANARY_TOKEN}}. Abstain with \"{{ABSTENTION_PHRASE}}\". "
    "Masks{{MASKED_PLACEHOLDERS}}."
)


def _override_registry(version: str = "2", prefix: str = "OVERRIDE") -> PromptRegistry:
    return PromptRegistry([PromptTemplate(SYSTEM_PROMPT_NAME, version, f"{prefix}\n{_MARKERS}")])


def _engine(llm, **kwargs) -> RAGEngine:
    chunk = make_chunk("d1")
    return RAGEngine(
        FakeEmbeddings(), FakeVectorStore([chunk]), llm, retriever=FakeRetriever([chunk]),
        **kwargs,
    )


def test_builtin_template_is_registered_under_its_name_and_version():
    template = builtin_prompts().get(SYSTEM_PROMPT_NAME)
    assert template.version == SYSTEM_PROMPT_TEMPLATE_VERSION
    assert SYSTEM_PROMPT_REF == template.ref
    assert render_system_prompt().text == SYSTEM_PROMPT


def test_a_registry_override_replaces_the_template_and_keeps_layers_and_canary():
    rendered = render_system_prompt(prompts=_override_registry(), persona="PIRATE")
    assert rendered.text.startswith("OVERRIDE\nPIRATE")
    assert CANARY_TOKEN in rendered.text
    assert rendered.ref.version == "2"


def test_an_override_without_every_marker_fails_to_render():
    registry = PromptRegistry([PromptTemplate(SYSTEM_PROMPT_NAME, "2", "{{PERSONA}} only")])
    with pytest.raises(PromptRenderError, match="unexpected variables"):
        render_system_prompt(prompts=registry)


def test_pinning_the_builtin_version_through_a_user_registry():
    registry = _override_registry()
    chained = PromptRegistry.chain(registry, builtin_prompts())
    chained.pin(SYSTEM_PROMPT_NAME, SYSTEM_PROMPT_TEMPLATE_VERSION)
    assert render_system_prompt(prompts=chained).text == SYSTEM_PROMPT


def test_engine_default_reports_the_builtin_ref():
    engine = _engine(FakeLLM())
    assert engine.get_system_prompt() == SYSTEM_PROMPT
    assert engine.get_system_prompt_ref() == SYSTEM_PROMPT_REF


def test_engine_renders_a_registry_override_and_keeps_its_version_label_separate():
    llm = FakeLLM()
    engine = _engine(llm, prompts=_override_registry(), system_prompt_version="prod-7")
    answer = engine.answer("query")

    assert llm.last_system.startswith("OVERRIDE")
    assert engine.get_system_prompt_ref().version == "2"
    # The deployment label (Answer/audit/cache key) is untouched by the registry.
    assert answer.system_prompt_version == "prod-7"


def test_explicit_text_wins_over_the_registry_and_is_recorded_as_custom():
    engine = _engine(FakeLLM(), system_prompt="my prompt", prompts=_override_registry())
    assert engine.get_system_prompt() == "my prompt"
    assert engine.get_system_prompt_ref() == PromptRef.for_text(SYSTEM_PROMPT_NAME, "my prompt")


def test_update_system_prompt_accepts_text_or_a_rendered_prompt():
    engine = _engine(FakeLLM())
    engine.update_system_prompt("hot reloaded")
    assert engine.get_system_prompt_ref().version == CUSTOM_VERSION

    rendered = render_system_prompt(prompts=_override_registry("5"))
    engine.update_system_prompt(rendered)
    assert engine.get_system_prompt() == rendered.text
    assert engine.get_system_prompt_ref().version == "5"

    with pytest.raises(TypeError):
        engine.update_system_prompt(42)  # type: ignore[arg-type]


def test_direct_mutation_of_the_prompt_text_is_reported_as_custom():
    engine = _engine(FakeLLM())
    engine._system_prompt = "mutated in place"  # the documented hot-reload path
    ref = engine.get_system_prompt_ref()
    assert ref.version == CUSTOM_VERSION
    assert ref == PromptRef.for_text(SYSTEM_PROMPT_NAME, "mutated in place")


def test_generate_span_and_audit_record_carry_the_prompt_ref(tmp_path):
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    exporter.clear()
    audit_path = tmp_path / "audit.jsonl"
    engine = _engine(FakeLLM(), prompts=_override_registry("9"), audit_log_path=str(audit_path))

    engine.answer("query")

    ref = engine.get_system_prompt_ref()
    [span] = [s for s in exporter.get_finished_spans() if s.name == "generate"]
    assert span.attributes["kuhaku.prompt.system.name"] == SYSTEM_PROMPT_NAME
    assert span.attributes["kuhaku.prompt.system.version"] == "9"
    assert span.attributes["kuhaku.prompt.system.hash"] == ref.content_hash

    row = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["metadata"]["prompts"]["system"] == ref.as_dict()
    # Provenance only: the prompt text itself never reaches the audit trail.
    assert "OVERRIDE" not in audit_path.read_text(encoding="utf-8")


def test_rag_facade_threads_the_registry_and_layers(monkeypatch):
    import kuhaku

    llm = FakeLLM()
    monkeypatch.setattr(kuhaku, "build_embedding_provider", lambda rs: FakeEmbeddings())
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: FakeVectorStore())
    monkeypatch.setattr(
        "kuhaku.tools.rag.resolve.adapters.embedding._model_cached", lambda name: True
    )
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s, **_k: llm)
    settings = Settings(_env_file=None, audit_enabled=False)

    rag = kuhaku.RAG(settings=settings, cache=False, prompts=_override_registry("3"))
    assert rag.engine.get_system_prompt().startswith("OVERRIDE")
    assert rag.engine.get_system_prompt_ref().version == "3"

    layered = kuhaku.RAG(
        settings=settings, cache=False, prompts=_override_registry("4"), persona="PIRATE"
    )
    assert layered.engine.get_system_prompt().startswith("OVERRIDE\nPIRATE")
    assert layered.engine.get_system_prompt_ref().version == "4"

    plain = kuhaku.RAG(settings=settings, cache=False, persona="PIRATE")
    assert plain.engine.get_system_prompt_ref() == SYSTEM_PROMPT_REF
