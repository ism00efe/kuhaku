"""Extractor([...]): picking a schema, then extracting with that schema's own provider."""

from __future__ import annotations

import logging

import pytest

from kuhaku.core.exceptions import ConfigError, CustomComponentError
from kuhaku.core.llm.base import LLMUnavailableError
from kuhaku.tools.extraction import (
    EvidenceStep,
    ExtractionSchema,
    ExtractionStatus,
    Extractor,
    RouteRequest,
)

from .conftest import Complaint, Order, ScriptedLLM, down


def pick(name):
    """The router LLM's reply: routing is an extraction of one ``choice`` field."""
    return {"data": {"choice": name}}


def _router(router_llm: ScriptedLLM, order_llm=None, complaint_llm=None, routes=None, **kwargs):
    router = Extractor(
        [
            Extractor(Order, llm=order_llm or ScriptedLLM(name="order:m")),
            Extractor(Complaint, llm=complaint_llm or ScriptedLLM(name="complaint:m")),
        ],
        llm=router_llm,
        **kwargs,
    )
    if routes is not None:
        router.routes[:] = routes
    return router


def test_each_schema_is_extracted_by_its_own_provider():
    order_llm = ScriptedLLM({"data": {"product": "tea", "quantity": 5}}, name="cheap:m")
    complaint_llm = ScriptedLLM({"data": {"topic": "late delivery"}}, name="strong:m")
    router_llm = ScriptedLLM(pick("Order"), pick("Complaint"), name="router:m")
    router = _router(router_llm, order_llm, complaint_llm)

    order = router.extract("5 tea please")
    complaint = router.extract("my delivery was late")

    assert (order.schema_name, order.ok) == ("Order", True)
    assert (complaint.schema_name, complaint.ok) == ("Complaint", True)
    assert (len(router_llm.calls), len(order_llm.calls), len(complaint_llm.calls)) == (2, 1, 1)


def test_the_llm_route_sees_every_schema_name_and_description():
    router_llm = ScriptedLLM(pick(None))
    _router(router_llm).route("hello")

    system, _ = router_llm.calls[0]
    assert "- Order: A customer placing an order." in system
    assert "- Complaint: A customer complaining about something." in system


def test_a_rule_in_front_of_the_llm_saves_the_llm_call():
    router_llm = ScriptedLLM(name="router:m")
    order_llm = ScriptedLLM({"data": {"product": "tea", "quantity": 5}})
    router = _router(router_llm, order_llm)

    def kilo_rule(request: RouteRequest) -> str | None:
        return "Order" if "kilo" in request.text else None

    router.routes.insert(0, kilo_rule)

    assert router.extract("5 kilo tea").schema_name == "Order"
    assert router_llm.calls == []


def test_no_matching_schema_is_no_match_never_a_default():
    result = _router(ScriptedLLM(pick(None))).extract("hello there")

    assert result.status is ExtractionStatus.NO_MATCH
    assert result.value is None
    assert result.trace_id


@pytest.mark.parametrize("reply", ["not json", pick("Invoice"), pick(["Order"]), pick(3)])
def test_an_unusable_route_reply_is_retried_then_defers_and_says_why(reply, caplog):
    router_llm = ScriptedLLM(reply, reply)
    with caplog.at_level(logging.WARNING, logger="kuhaku.tools.extraction.router"):
        result = _router(router_llm).extract("hello")

    assert result.status is ExtractionStatus.NO_MATCH
    assert len(router_llm.calls) == 2  # the retry loop asked again
    assert any("deferring" in record.getMessage() for record in caplog.records)


def test_a_retried_route_reply_routes():
    order_llm = ScriptedLLM({"data": {"product": "tea", "quantity": 5}})
    router = _router(ScriptedLLM(pick(["Order"]), pick("Order")), order_llm)
    assert router.extract("5 tea").schema_name == "Order"


def test_route_fallback_and_instructions():
    order_llm = ScriptedLLM({"data": {"product": "tea", "quantity": 5}})
    strong = ScriptedLLM(pick("Order"), name="strong:m")
    router = _router(
        ScriptedLLM("x", "x"),
        order_llm,
        fallback_llms=[strong],
        instructions="A bare quantity continues an order.",
    )

    assert router.extract("5").schema_name == "Order"
    system, _ = strong.calls[0]
    assert "A bare quantity continues an order." in system


def test_a_crashing_custom_route_is_named():
    def broken(request):
        raise KeyError("oops")

    with pytest.raises(CustomComponentError, match="broken"):
        _router(ScriptedLLM(), routes=[broken]).route("hello")


def test_a_custom_route_returning_a_non_string_is_a_component_error():
    with pytest.raises(CustomComponentError):
        _router(ScriptedLLM(), routes=[lambda request: ["Order"]]).route("hello")


def test_a_route_naming_an_unregistered_schema_is_a_component_error():
    router = _router(ScriptedLLM(), routes=[lambda request: "Invoice"])
    with pytest.raises(CustomComponentError):
        router.route("hello")


def test_router_provider_failure_raises():
    with pytest.raises(LLMUnavailableError):
        _router(ScriptedLLM(down())).extract("hello")


def test_prior_carries_over_only_to_the_same_schema():
    order_llm = ScriptedLLM(
        {"data": {"product": "tea", "quantity": None}},
        {"data": {"product": None, "quantity": 5}},
    )
    complaint_llm = ScriptedLLM({"data": {"topic": "late"}})
    router_llm = ScriptedLLM(pick("Order"), pick("Order"), pick("Complaint"))
    router = _router(router_llm, order_llm, complaint_llm)

    first = router.extract("tea")
    second = router.extract("5 kilo", prior=first)
    switched = router.extract("it was late", prior=second)

    assert second.value == Order(product="tea", quantity=5)
    assert switched.ok
    _, complaint_prompt = complaint_llm.calls[0]
    assert "earlier turns" not in complaint_prompt


def test_duplicate_schema_names_are_refused():
    with pytest.raises(ConfigError):
        Extractor(
            [
                Extractor(Order, llm=ScriptedLLM()),
                Extractor(ExtractionSchema(Complaint, name="Order"), llm=ScriptedLLM()),
            ]
        )


def test_an_empty_list_is_refused():
    with pytest.raises(TypeError):
        Extractor([])


def test_a_list_mixing_field_names_and_extractors_is_refused():
    with pytest.raises(TypeError, match="not both"):
        Extractor(["product", Extractor(Order, llm=ScriptedLLM())])


def test_options_of_one_extractor_are_refused_on_a_list():
    with pytest.raises(ConfigError, match="belong on the extractors"):
        Extractor([Extractor(Order, llm=ScriptedLLM())], steps=[EvidenceStep()])


def test_one_schema_has_no_route_and_a_list_has_no_preview():
    single = Extractor(Order, llm=ScriptedLLM())
    with pytest.raises(ConfigError):
        single.route("5 tea")
    with pytest.raises(ConfigError):
        Extractor([single], llm=ScriptedLLM()).preview("5 tea")


def test_routes_default_to_the_first_extractors_provider():
    order_llm = ScriptedLLM(pick("Order"), {"data": {"product": "tea", "quantity": 5}})
    router = Extractor([Extractor(Order, llm=order_llm, enable_token_tracking=False)])

    assert router.llm is order_llm
    assert router.extract("5 tea").ok


def test_router_refuses_a_bare_dict_as_prior():
    router = _router(ScriptedLLM())  # any call fails the test
    with pytest.raises(TypeError, match="ExtractionResult"):
        router.extract("5 tea", prior={"product": "tea"})


def test_an_empty_text_is_no_match_without_a_model_call():
    router = _router(ScriptedLLM())  # any call fails the test
    assert router.extract("  ").status is ExtractionStatus.NO_MATCH


def test_a_non_str_text_is_the_callers_type_error():
    router = _router(ScriptedLLM())
    with pytest.raises(TypeError, match="text must be a str"):
        router.extract(None)


def test_an_interrupted_order_is_picked_up_after_a_complaint_and_a_greeting():
    order_llm = ScriptedLLM(
        {"data": {"product": "tea", "quantity": None}},
        {"data": {"product": None, "quantity": 5}},
    )
    complaint_llm = ScriptedLLM({"data": {"topic": "late"}})
    router_llm = ScriptedLLM(pick("Order"), pick("Complaint"), pick(None), pick("Order"))
    router = _router(router_llm, order_llm, complaint_llm)

    result = None
    for message in ["tea", "it was late", "hello", "5 kilo"]:
        result = router.extract(message, prior=result)

    assert result.schema_name == "Order"
    assert result.value == Order(product="tea", quantity=5)
    # The complaint's data stays aside for when the conversation comes back to it.
    assert result.metadata["earlier"] == {"Complaint": {"topic": "late"}}
