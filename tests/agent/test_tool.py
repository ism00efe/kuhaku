"""AgentTool: schemas, argument binding, entitlement. No model involved."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from kuhaku import AuthContext
from kuhaku.tools.agent import AgentTool, ArgumentsInvalid, is_entitled


def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Current weather for a city.

    Not part of the description.
    """

    return f"12{unit} in {city}"


def whoami(greeting: str, auth_context: AuthContext | None = None) -> str:
    """Greets the caller."""

    return f"{greeting} {auth_context.identity if auth_context else 'nobody'}"


class Refund(BaseModel):
    """Refund part of an order."""

    order_id: str
    amount: float


def test_schema_from_function():
    tool = AgentTool(get_weather)
    assert tool.name == "get_weather"
    assert tool.spec.description == "Current weather for a city."
    assert tool.spec.parameters["required"] == ["city"]
    assert tool.spec.parameters["properties"]["unit"]["enum"] == ["c", "f"]


def test_auth_context_is_left_out_of_the_schema_and_cannot_be_sent():
    tool = AgentTool(whoami, pass_auth_context=True)
    assert "auth_context" not in tool.spec.parameters["properties"]
    with pytest.raises(ArgumentsInvalid, match="unexpected arguments: auth_context"):
        tool.bind({"greeting": "hi", "auth_context": {"identity": "root"}})
    args, kwargs = tool.bind({"greeting": "hi"})
    assert tool.invoke(args, kwargs, AuthContext(identity="ada")) == "hi ada"


def test_pass_auth_context_needs_the_parameter():
    with pytest.raises(TypeError, match="auth_context"):
        AgentTool(get_weather, pass_auth_context=True)


def test_args_model_tool_gets_one_validated_instance():
    seen = []

    def refund(request: Refund) -> str:
        seen.append(request)
        return "ok"

    tool = AgentTool(refund, args_model=Refund)
    assert tool.spec.description == "Refund part of an order."
    assert set(tool.spec.parameters["properties"]) == {"order_id", "amount"}
    args, kwargs = tool.bind({"order_id": "42", "amount": "5"})
    assert tool.invoke(args, kwargs, None) == "ok"
    assert seen == [Refund(order_id="42", amount=5.0)]
    with pytest.raises(ArgumentsInvalid, match="unexpected arguments: note"):
        tool.bind({"order_id": "42", "amount": 5, "note": "x"})


def test_invalid_arguments_name_the_field_but_never_echo_the_value():
    tool = AgentTool(get_weather)
    with pytest.raises(ArgumentsInvalid) as info:
        tool.bind({"city": "Oslo", "unit": "SECRET-VALUE-123"})
    assert "unit" in str(info.value)
    assert "SECRET-VALUE-123" not in str(info.value)
    with pytest.raises(ArgumentsInvalid, match="city"):
        tool.bind({})


def test_values_are_coerced_by_the_schema():
    def add(a: int, b: int) -> int:
        return a + b

    tool = AgentTool(add)
    args, kwargs = tool.bind({"a": "2", "b": 3})
    assert tool.invoke(args, kwargs, None) == 5


def test_access_tags_are_validated():
    with pytest.raises(TypeError, match="single string"):
        AgentTool(get_weather, access_tags="admin")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        AgentTool(get_weather, access_tags=["  "])


def test_async_callables_and_bad_timeouts_are_rejected():
    async def fetch(url: str) -> str:
        return url

    with pytest.raises(TypeError, match="async"):
        AgentTool(fetch)
    with pytest.raises(ValueError, match="timeout"):
        AgentTool(get_weather, timeout=0)


@pytest.mark.parametrize(
    ("tags", "ctx", "visible"),
    [
        ((), None, True),
        ((), AuthContext(identity="ada"), True),
        (("hr",), None, False),
        (("hr",), AuthContext(identity="ada", roles=("eng",)), False),
        (("hr",), AuthContext(identity="ada", roles=("eng", "hr")), True),
        (("Admin",), AuthContext(identity="ada", roles=("admin",)), False),
    ],
)
def test_entitlement_matches_the_rag_rule(tags, ctx, visible):
    assert is_entitled(AgentTool(get_weather, access_tags=tags), ctx) is visible
