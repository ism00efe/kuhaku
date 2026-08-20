"""Tests for prompt assembly."""

from __future__ import annotations

from kuhaku.tools.rag.messages import EngineMessages
from kuhaku.tools.rag.models import RetrievedChunk
from kuhaku.core.security.guard import CANARY_TOKEN
from kuhaku.tools.rag.prompts import (
    ABSTENTION_PHRASE,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from tests.conftest import make_chunk


def test_system_prompt_is_turkish_and_grounded():
    assert "Türkçe" in SYSTEM_PROMPT
    assert "[S1]" in SYSTEM_PROMPT  # instructs citation format


def test_system_prompt_instructs_exact_abstention_phrase():
    assert ABSTENTION_PHRASE in SYSTEM_PROMPT


def test_build_user_prompt_numbers_sources():
    retrieved = [
        RetrievedChunk(make_chunk("d1", text="first source"), 0.9),
        RetrievedChunk(make_chunk("d2", 1, text="second source"), 0.8),
    ]
    prompt = build_user_prompt("why declined?", retrieved)
    assert "QUESTION:" in prompt and "why declined?" in prompt
    assert "[S1]" in prompt and "[S2]" in prompt
    assert "first source" in prompt and "second source" in prompt


def test_build_user_prompt_handles_no_sources():
    prompt = build_user_prompt("question", [])
    assert "No sources found" in prompt


def test_build_user_prompt_uses_injected_question_and_sources_labels():
    """EngineMessages.question_label/sources_label (renamed from the Turkish
    soru_label/kaynaklar_label) drive the prompt's section headers."""

    custom = EngineMessages(question_label="Q:", sources_label="SRC:")
    retrieved = [RetrievedChunk(make_chunk("d1", text="first source"), 0.9)]

    prompt = build_user_prompt("why declined?", retrieved, custom)

    assert "Q:\nwhy declined?" in prompt
    assert "SRC:\n" in prompt
    assert "QUESTION:" not in prompt and "SOURCES:" not in prompt


def test_build_user_prompt_single_source_has_no_dangling_s2_reference():
    """Regression: a 1-chunk retrieval must never put an unused [S2] into the prompt --
    see DECISIONS.md D38 (the latent build_user_prompt bug FR4 forced into scope)."""

    retrieved = [RetrievedChunk(make_chunk("d1", text="only source"), 0.9)]
    prompt = build_user_prompt("soru", retrieved)
    assert "[S1]" in prompt
    assert "[S2]" not in prompt


# --- FR1: system prompt content (Category 2) ------------------------------------
def test_system_prompt_covers_payment_domain_terms():
    for term in ("FAST", "EFT", "SWIFT", "PCI DSS"):
        assert term in SYSTEM_PROMPT


def test_system_prompt_instructs_contradiction_handling():
    assert "Contradictions" in SYSTEM_PROMPT


def test_system_prompt_includes_a_one_shot_example():
    assert "## Example" in SYSTEM_PROMPT
    assert "SORU:" in SYSTEM_PROMPT and "KAYNAKLAR:" in SYSTEM_PROMPT


def test_system_prompt_instructs_turkish_output_in_english_control_layer():
    """FR1: the control-layer instructions are in English; only the output-language and
    abstention requirements are stated in/around Turkish."""

    assert "Always respond in Turkish" in SYSTEM_PROMPT


# --- guard v2: prompt hardening (D39) ---------------------------------------------
# NOTE: the "prompt version bumped to v2 for the hardened template" assertion moved to
# tests/rag/test_config.py (test_ragsettings_defaults_cover_the_fields_moved_from_settings)
# -- PROMPT_VERSION no longer lives in this module, see DECISIONS.md D42.
def test_system_prompt_declares_instruction_precedence():
    assert "Instruction precedence" in SYSTEM_PROMPT


def test_system_prompt_declares_data_marking():
    assert "Data marking" in SYSTEM_PROMPT
    assert "[DOC]" in SYSTEM_PROMPT and "[/DOC]" in SYSTEM_PROMPT


def test_system_prompt_contains_the_substituted_canary_token():
    """The {{CANARY_TOKEN}} placeholder must be substituted, not left literal, and must
    match the actual runtime canary the output guard checks for."""

    assert "{{CANARY_TOKEN}}" not in SYSTEM_PROMPT
    assert CANARY_TOKEN in SYSTEM_PROMPT


def test_build_user_prompt_wraps_each_source_body_in_doc_markers():
    retrieved = [RetrievedChunk(make_chunk("d1", text="first source"), 0.9)]
    prompt = build_user_prompt("soru", retrieved)
    assert "[DOC]\nfirst source\n[/DOC]" in prompt


def test_build_user_prompt_doc_wrapping_preserves_the_dangling_s2_fix():
    """The [DOC] reformat must not reintroduce the D38 dangling-[S2]-on-1-chunk bug."""

    retrieved = [RetrievedChunk(make_chunk("d1", text="only source"), 0.9)]
    prompt = build_user_prompt("soru", retrieved)
    assert "[S1]" in prompt
    assert "[S2]" not in prompt
