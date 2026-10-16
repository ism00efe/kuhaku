"""Seeing and replacing the prompt texts the tool sends."""

from __future__ import annotations

import pytest

from kuhaku.core.exceptions import ConfigError
from kuhaku.tools.extraction import (
    FIXED_RULES,
    EvidenceStep,
    ExtractionPrompts,
    ExtractionRouter,
    Extractor,
    LLMRoute,
)

from .conftest import Complaint, Order, ScriptedLLM

GOOD = {"data": {"product": "tea", "quantity": 5}}


def test_preview_shows_the_prompt_without_calling_the_model():
    llm = ScriptedLLM()  # no replies scripted: any call would fail the test
    system, user = Extractor(Order, llm=llm, steps=[EvidenceStep()]).preview("5 tea")

    assert llm.calls == []
    assert FIXED_RULES in system
    assert "You extract structured data" in system
    assert "add an entry to \"evidence\"" in system  # the evidence step's contribution
    assert "5 tea" in user


def test_preview_matches_what_is_sent():
    llm = ScriptedLLM(GOOD)
    extractor = Extractor(Order, llm=llm)
    system, _ = extractor.preview("5 tea")
    extractor.extract("5 tea")
    assert llm.calls[0][0] == system


def test_rules_are_replaced_in_one_line_and_fixed_rules_stay():
    llm = ScriptedLLM(GOOD)
    Extractor(Order, llm=llm, prompts={"rules": "Sen bir sipariş asistanısın."}).extract("x")

    system, _ = llm.calls[0]
    assert system.startswith("Sen bir sipariş asistanısın.")
    assert "You extract structured data" not in system
    assert FIXED_RULES in system


def test_retry_evidence_and_slot_filling_texts_are_replaceable():
    llm = ScriptedLLM(
        "broken",
        {"data": {"product": None, "quantity": 5}, "evidence": {"/quantity": "5"}},
    )
    prompts = {
        "retry": "ÖNCEKİ: {previous_output} SORUN: {problems}",
        "evidence": "KANIT VER",
        "slot_filling": "ÖNCEKİ TURLAR",
    }
    Extractor(Order, llm=llm, steps=[EvidenceStep()], prompts=prompts).extract(
        "5 kilo", prior={"product": "çay"}
    )

    first_system, first_user = llm.calls[0]
    assert "KANIT VER" in first_system
    assert "ÖNCEKİ TURLAR" in first_user
    _, retry_user = llm.calls[1]
    assert "ÖNCEKİ: broken SORUN: - " in retry_user


def test_route_text_is_replaceable():
    router_llm = ScriptedLLM({"data": {"choice": "Order"}})
    router = ExtractionRouter(
        [Extractor(Order, llm=ScriptedLLM(GOOD)), Extractor(Complaint, llm=ScriptedLLM())],
        llm=router_llm,
        routes=[LLMRoute(prompts={"route": "ŞEMALAR:\n{schemas}"})],
    )
    router.extract("5 tea")

    system, _ = router_llm.calls[0]
    assert "ŞEMALAR:\n- Order: A customer placing an order." in system


@pytest.mark.parametrize(
    "prompts",
    [
        {"retry": "no placeholders"},
        {"route": "no schemas placeholder"},
        {"rulez": "typo"},
        {"rules": 5},
    ],
)
def test_bad_prompt_overrides_are_refused(prompts):
    with pytest.raises(ConfigError):
        Extractor(Order, llm=ScriptedLLM(), prompts=prompts)


def test_braces_in_a_custom_template_are_kept():
    prompts = ExtractionPrompts(retry='{"hint": 1} {previous_output} {problems}')
    llm = ScriptedLLM("broken", GOOD)
    Extractor(Order, llm=llm, prompts=prompts).extract("5 tea")
    assert '{"hint": 1} broken' in llm.calls[1][1]
