"""Tests for FR4's output guard: citation grounding, canary detection, PII egress.

The guard itself is tool-agnostic (plain citation tags + context strings, no RAG type) --
these tests build those generic shapes directly rather than via RAG's Citation/
RetrievedChunk.
"""

from __future__ import annotations

from kuhaku.core.security.output_guard import (
    SAFE_FALLBACK,
    _iban_ok,
    annotate_ungrounded_citations,
    check_citation_grounding,
    check_pii_egress,
    detect_canary,
    evaluate_output,
)


def _contexts(*texts: str) -> list[str]:
    return list(texts)


# --- citation grounding (item 8) --------------------------------------------------
def test_high_overlap_citation_is_not_annotated():
    contexts = _contexts("3D Secure timeout occurs when the issuer does not respond in time.")
    text = "3D Secure timeout occurs when the issuer does not respond in time. [S1]"
    ungrounded = check_citation_grounding(text, ["S1"], contexts, threshold=0.1)
    assert ungrounded == []


def test_low_overlap_citation_is_flagged():
    contexts = _contexts("3D Secure timeout occurs when the issuer does not respond in time.")
    text = "Kartınızı iptal etmeniz gerekiyor. [S1]"
    ungrounded = check_citation_grounding(text, ["S1"], contexts, threshold=0.1)
    assert ungrounded == ["S1"]


def test_annotate_appends_marker_after_flagged_tag():
    text = "You need to cancel your card. [S1]"
    annotated = annotate_ungrounded_citations(text, ["S1"])
    assert annotated == "You need to cancel your card. [S1][unverified]"


def test_annotate_is_noop_for_no_flagged_tags():
    text = "Some grounded answer. [S1]"
    assert annotate_ungrounded_citations(text, []) == text


def test_out_of_range_citation_is_skipped_not_flagged():
    contexts = _contexts("some source text")
    # tag S5 refers to a source index beyond what was retrieved -- the calling tool's own
    # unverified-citation flagging handles this, output_guard should ignore it.
    ungrounded = check_citation_grounding("text [S5]", ["S5"], contexts, threshold=0.9)
    assert ungrounded == []


# --- canary detection (item 9) ------------------------------------------------------
def test_full_canary_token_is_detected():
    canary = "abcdef1234567890"
    assert detect_canary(f"the secret is {canary}", canary) is True


def test_canary_substring_of_at_least_8_chars_is_detected():
    canary = "abcdef1234567890"
    assert detect_canary("leaked: abcdef12 partial", canary) is True


def test_short_substring_below_8_chars_is_not_detected():
    canary = "abcdef1234567890"
    assert detect_canary("mentions abcdef only", canary) is False


def test_unrelated_text_has_no_canary():
    canary = "abcdef1234567890"
    assert detect_canary("Ödeme başarısız oldu, tekrar deneyin.", canary) is False


# --- PII egress (item 10) -----------------------------------------------------------
def test_iban_checksum_validator():
    # A real, checksum-valid Turkish IBAN (structurally valid, not a real account).
    assert _iban_ok("TR330006100519786457841326") is True
    assert _iban_ok("TR000000000000000000000000") is False


def test_valid_iban_not_in_chunks_is_flagged():
    contexts = _contexts("Genel ödeme süreci hakkında bilgi.")
    text = "IBAN numaranız: TR330006100519786457841326"
    categories = check_pii_egress(text, contexts)
    assert categories == ["IBAN"]


def test_valid_pan_not_in_chunks_is_flagged():
    contexts = _contexts("Genel ödeme süreci hakkında bilgi.")
    text = "Kart numaranız: 4111 1111 1111 1111"
    categories = check_pii_egress(text, contexts)
    assert categories == ["PAN"]


def test_valid_national_id_not_in_chunks_is_flagged():
    contexts = _contexts("Genel ödeme süreci hakkında bilgi.")
    text = "TC kimlik numaranız: 10000000146"
    categories = check_pii_egress(text, contexts)
    assert categories == ["NATIONAL_ID"]


def test_pii_present_verbatim_in_a_chunk_is_not_flagged():
    contexts = _contexts("Test card 4111 1111 1111 1111 is used in the sandbox.")
    text = "Use test card 4111 1111 1111 1111 for sandbox testing."
    categories = check_pii_egress(text, contexts)
    assert categories == []


def test_invalid_luhn_number_is_not_flagged():
    contexts = _contexts("no pii here")
    text = "reference number: 1234 5678 9012 3456"  # fails Luhn
    categories = check_pii_egress(text, contexts)
    assert categories == []


# --- evaluate_output composition -----------------------------------------------------
def test_evaluate_output_blocks_on_canary_and_ignores_pii_check():
    canary = "abcdef1234567890"
    contexts = _contexts("some source")
    text = f"leaked token {canary}, also card 4111 1111 1111 1111"
    result = evaluate_output(text, [], contexts, canary=canary, citation_grounding_threshold=0.1)
    assert result.blocked is True
    assert result.canary_detected is True
    assert result.text == SAFE_FALLBACK
    # PII check never runs once canary already fired.
    assert result.pii_egress_detected is False


def test_evaluate_output_blocks_on_pii_egress():
    contexts = _contexts("some unrelated source")
    text = "Kart numaranız: 4111 1111 1111 1111"
    result = evaluate_output(
        text, [], contexts, canary="unrelated-canary-value", citation_grounding_threshold=0.1
    )
    assert result.blocked is True
    assert result.pii_egress_detected is True
    assert result.pii_egress_categories == ["PAN"]
    assert result.text == SAFE_FALLBACK


def test_evaluate_output_passes_through_a_clean_grounded_answer():
    contexts = _contexts("3D Secure timeout occurs when the issuer does not respond in time.")
    text = "3D Secure timeout occurs when the issuer does not respond in time. [S1]"
    result = evaluate_output(
        text, ["S1"], contexts,
        canary="unrelated-canary", citation_grounding_threshold=0.1,
    )
    assert result.blocked is False
    assert result.text == text
    assert result.ungrounded_citations == []


def test_evaluate_output_fails_open_on_internal_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "kuhaku.core.security.output_guard.check_citation_grounding", _boom
    )
    text = "Some answer text. [S1]"
    result = evaluate_output(text, [], [], canary="c", citation_grounding_threshold=0.1)
    assert result.blocked is False
    assert result.text == text
