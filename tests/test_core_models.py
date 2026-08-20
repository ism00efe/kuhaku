"""Tests for the generic core domain models (kuhaku.core.models).

These are tool-agnostic primitives -- no RAG type (Document, Chunk, ...) belongs here;
see tests/rag/test_models.py for those.
"""

from __future__ import annotations

import pytest

from kuhaku.core.models import ExecutionResult, Message, ToolCall


def test_message_holds_role_and_content():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.metadata == {}


def test_message_metadata_defaults_to_independent_empty_dicts():
    a = Message(role="user", content="a")
    b = Message(role="user", content="b")
    a.metadata["x"] = 1
    assert b.metadata == {}


def test_tool_call_holds_name_and_arguments():
    call = ToolCall(name="search", arguments={"query": "refund policy"})
    assert call.name == "search"
    assert call.arguments == {"query": "refund policy"}


def test_tool_call_arguments_default_to_empty_dict():
    call = ToolCall(name="noop")
    assert call.arguments == {}


def test_execution_result_defaults_to_success():
    result = ExecutionResult(output="ok")
    assert result.success is True
    assert result.error is None
    assert result.metadata == {}


def test_execution_result_can_represent_a_failure():
    result = ExecutionResult(output=None, success=False, error="timeout")
    assert result.success is False
    assert result.error == "timeout"


# --- the moved RAG models are no longer importable from here --------------------
def test_rag_models_are_not_importable_from_core_models():
    """The re-export shim was removed entirely; these names now live only under
    kuhaku.tools.rag.models -- accessing them here is a plain AttributeError, not a
    deprecation warning."""

    import kuhaku.core.models as core_models  # noqa: PLC0415

    for name in ("Document", "Chunk", "RetrievedChunk", "Citation", "Answer"):
        with pytest.raises(AttributeError):
            getattr(core_models, name)


def test_unknown_attribute_still_raises_attribute_error():
    import kuhaku.core.models as core_models  # noqa: PLC0415

    with pytest.raises(AttributeError):
        core_models.NotARealModel  # noqa: B018
