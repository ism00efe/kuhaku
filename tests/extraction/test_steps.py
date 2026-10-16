"""The optional steps: slot filling, evidence, validation -- and the pipeline holding them."""

from __future__ import annotations

import pytest

from kuhaku.core.exceptions import ConfigError, CustomComponentError
from kuhaku.tools.extraction import (
    EvidenceStep,
    ExtractionStatus,
    Extractor,
    ExtractStep,
    Issue,
    IssueKind,
    Pipeline,
    Provenance,
    Step,
    ValidationStep,
)
from kuhaku.tools.extraction.steps.slot_filling import merge

from .conftest import Order, ScriptedLLM

# --- pipeline -------------------------------------------------------------------------


class Marker(Step):
    name = "marker"

    def __init__(self) -> None:
        self.ran = False

    def run(self, ctx) -> None:
        self.ran = True
        ctx.metadata["marker"] = True


def test_a_pipeline_needs_an_extract_step():
    with pytest.raises(ConfigError):
        Pipeline([Marker()])


def test_the_extract_step_cannot_be_removed_or_disabled_only_replaced():
    pipeline = Pipeline([ExtractStep()])
    with pytest.raises(ConfigError):
        pipeline.remove("extract")
    with pytest.raises(ConfigError):
        pipeline.disable("extract")

    replacement = ExtractStep(max_retries=5)
    pipeline.replace("extract", replacement)
    assert pipeline.get("extract") is replacement


def test_steps_can_be_added_positioned_disabled_and_removed():
    pipeline = Pipeline([ExtractStep()])
    pipeline.add(Marker(), before="extract")
    assert pipeline.names == ["marker", "extract"]

    pipeline.disable("marker")
    assert not pipeline.is_enabled("marker")
    assert "marker (disabled)" in repr(pipeline)

    pipeline.remove("marker")
    assert pipeline.names == ["extract"]


def test_duplicate_and_unnamed_steps_are_refused():
    pipeline = Pipeline([ExtractStep()])
    with pytest.raises(ConfigError):
        pipeline.add(ExtractStep())
    with pytest.raises(ConfigError):
        pipeline.add(Step())
    with pytest.raises(ConfigError):
        pipeline.replace("extract", Marker())


def test_a_disabled_step_does_not_run():
    marker = Marker()
    extractor = Extractor(
        Order, llm=ScriptedLLM({"data": {"product": "tea", "quantity": 5}}), steps=[marker]
    )
    extractor.pipeline.disable("marker")

    result = extractor.extract("5 tea")

    assert not marker.ran
    assert "marker" not in result.metadata


def test_steps_after_a_failed_extract_with_no_data_do_not_run():
    marker = Marker()
    Extractor(Order, llm=ScriptedLLM("x", "x"), steps=[marker]).extract("tea")
    assert not marker.ran


# --- slot filling ---------------------------------------------------------------------


def test_a_follow_up_completes_the_earlier_turn():
    llm = ScriptedLLM(
        {"data": {"product": "çay", "quantity": None}},
        {"data": {"product": None, "quantity": 5}},
    )
    extractor = Extractor(Order, llm=llm)

    first = extractor.extract("çay yaz")
    second = extractor.extract("5 kilo", prior=first)

    assert first.status is ExtractionStatus.NEEDS_CLARIFICATION
    assert second.status is ExtractionStatus.COMPLETE
    assert second.value == Order(product="çay", quantity=5)
    _, user = llm.calls[1]
    assert "Values already extracted in earlier turns" in user


def test_prior_as_a_plain_dict_works_too():
    llm = ScriptedLLM({"data": {"product": None, "quantity": 5}})
    result = Extractor(Order, llm=llm).extract("5 kilo", prior={"product": "çay"})
    assert result.value == Order(product="çay", quantity=5)


def test_without_prior_slot_filling_is_inert():
    llm = ScriptedLLM({"data": {"product": "tea", "quantity": 5}})
    Extractor(Order, llm=llm).extract("5 tea")
    _, user = llm.calls[0]
    assert "earlier turns" not in user


def test_a_new_object_after_a_finished_one_carries_nothing_over():
    llm = ScriptedLLM(
        {"data": {"product": "tea", "quantity": 3}},
        {"data": {"product": "coffee", "quantity": None}, "new": True},
    )
    extractor = Extractor(Order, llm=llm)

    first = extractor.extract("3 boxes of tea")
    second = extractor.extract("I'd like coffee", prior=first)

    assert second.status is ExtractionStatus.NEEDS_CLARIFICATION
    assert second.data == {"product": "coffee", "quantity": None}
    assert second.missing_slots == ("/quantity",)
    _, user = llm.calls[1]
    assert '"new"' in user


def test_a_correction_of_a_finished_object_keeps_the_rest():
    llm = ScriptedLLM(
        {"data": {"product": "tea", "quantity": 3}},
        {"data": {"product": None, "quantity": 4}, "new": False},
    )
    extractor = Extractor(Order, llm=llm)

    second = extractor.extract("make it 4", prior=extractor.extract("3 boxes of tea"))

    assert second.value == Order(product="tea", quantity=4)


def test_without_prior_the_model_is_not_asked_about_a_new_object():
    llm = ScriptedLLM({"data": {"product": "tea", "quantity": 5}})
    Extractor(Order, llm=llm).extract("5 tea")
    _, user = llm.calls[0]
    assert '"new"' not in user


def test_merge_rules():
    assert merge({"a": 1, "b": 2}, {"a": None, "b": 3}) == {"a": 1, "b": 3}
    assert merge({"x": {"y": 1, "z": 2}}, {"x": {"z": 9}}) == {"x": {"y": 1, "z": 9}}
    assert merge({"items": [1]}, {"items": []}) == {"items": [1]}
    assert merge({"items": [1]}, {"items": [2, 3]}) == {"items": [2, 3]}


# --- evidence -------------------------------------------------------------------------


def test_evidence_found_in_the_source_is_grounded():
    llm = ScriptedLLM(
        {
            "data": {"product": "tea", "quantity": 5},
            "evidence": {"/product": "TEA", "/quantity": "5  boxes"},
        }
    )
    result = Extractor(Order, llm=llm, steps=[EvidenceStep()]).extract("5 boxes of tea")

    assert result.ok
    assert result.fields["/product"].provenance is Provenance.SOURCE
    assert result.fields["/quantity"].provenance is Provenance.SOURCE
    system, user = llm.calls[0]
    assert '"evidence"' in user and "JSON Pointer" in system


def test_an_invented_quote_is_ungrounded_and_only_reported_by_default():
    llm = ScriptedLLM(
        {"data": {"product": "coffee", "quantity": 5}, "evidence": {"/product": "coffee"}}
    )
    result = Extractor(Order, llm=llm, steps=[EvidenceStep()]).extract("5 tea")

    assert result.ok
    assert result.fields["/product"].provenance is Provenance.UNGROUNDED
    assert result.fields["/quantity"].provenance is Provenance.UNGROUNDED


def test_on_ungrounded_invalid_rejects_the_result():
    llm = ScriptedLLM(
        {"data": {"product": "coffee", "quantity": 5}, "evidence": {"/quantity": "5"}}
    )
    step = EvidenceStep(on_ungrounded="invalid")
    result = Extractor(Order, llm=llm, steps=[step]).extract("5 tea")

    assert result.status is ExtractionStatus.INVALID
    assert [issue.path for issue in result.issues] == ["/product"]


def test_values_carried_from_prior_are_marked_prior():
    llm = ScriptedLLM({"data": {"product": None, "quantity": 5}, "evidence": {"/quantity": "5"}})
    result = Extractor(Order, llm=llm, steps=[EvidenceStep()]).extract(
        "5 kilo", prior={"product": "çay"}
    )

    assert result.fields["/product"].provenance is Provenance.PRIOR
    assert result.fields["/quantity"].provenance is Provenance.SOURCE


def test_unknown_on_ungrounded_mode_is_refused():
    with pytest.raises(ConfigError):
        EvidenceStep(on_ungrounded="ignore")


# --- validation -----------------------------------------------------------------------


def _at_most_100(order, ctx):
    if order.quantity > 100:
        yield Issue(IssueKind.INVALID, "too_much", "at most 100", "/quantity")


def test_a_caller_rule_turns_a_valid_extraction_invalid():
    llm = ScriptedLLM({"data": {"product": "tea", "quantity": 500}})
    result = Extractor(Order, llm=llm, steps=[ValidationStep(_at_most_100)]).extract("500 tea")

    assert result.status is ExtractionStatus.INVALID
    assert result.value is None
    assert result.data == {"product": "tea", "quantity": 500}


def test_a_passing_rule_leaves_the_result_complete():
    llm = ScriptedLLM({"data": {"product": "tea", "quantity": 5}})
    assert Extractor(Order, llm=llm, steps=[ValidationStep(_at_most_100)]).extract("5 tea").ok


def test_a_rule_returning_a_retryable_issue_is_a_component_error():
    def bad(order, ctx):
        return [Issue(IssueKind.RETRYABLE, "x", "x")]

    llm = ScriptedLLM({"data": {"product": "tea", "quantity": 5}})
    with pytest.raises(CustomComponentError):
        Extractor(Order, llm=llm, steps=[ValidationStep(bad)]).extract("5 tea")


def test_a_non_callable_rule_is_refused():
    with pytest.raises(CustomComponentError):
        ValidationStep("not a function")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "quantity"),
    [("1.500 boxes of tea", 1500), ("1,500 boxes of tea", 1500), ("1 500 boxes of tea", 1500),
     ("2,5 boxes of tea", 2.5), ("1.500,5 boxes of tea", 1500.5), ("-3 boxes of tea", -3)],
)
def test_evidence_accepts_a_number_in_its_local_form(text, quantity):
    quote = text.removesuffix(" of tea")
    llm = ScriptedLLM(
        {"data": {"product": "tea", "quantity": quantity}, "evidence": {"/product": "tea", "/quantity": quote}}
    )
    result = Extractor(Order, llm=llm, steps=[EvidenceStep()]).extract(text)
    assert result.fields["/quantity"].provenance is Provenance.SOURCE


def test_evidence_does_not_read_a_grouped_number_as_a_decimal():
    llm = ScriptedLLM(
        {"data": {"product": "tea", "quantity": 1.5}, "evidence": {"/product": "tea", "/quantity": "1.500 boxes"}}
    )
    result = Extractor(Order, llm=llm, steps=[EvidenceStep()]).extract("1.500 boxes of tea")
    assert result.fields["/quantity"].provenance is Provenance.MISMATCH
