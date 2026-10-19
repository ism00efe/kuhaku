"""Regression tests for the classification parity findings: native structured output,
the prompt registry, response usage, sessions, attachments, history text, documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from kuhaku.core.auth import AuthContext
from kuhaku.core.exceptions import ConfigError
from kuhaku.core.llm.base import (
    LLMCapabilities,
    LLMResponse,
    StructuredOutputMode,
    TokenUsage,
    UnsupportedContentError,
)
from kuhaku.core.llm.provider import BaseLLMProvider
from kuhaku.core.media import ImagePart
from kuhaku.core.models import Message, TextPart
from kuhaku.core.prompts import PromptRegistry, PromptTemplate
from kuhaku.core.session import InMemorySessionStore
from kuhaku.tools.classification import ClassificationStatus, Classifier
from kuhaku.tools.classification.classifier import SESSION_NAMESPACE
from tests.test_media import tiny_png

from .conftest import LABELS, ScriptedLLM, answer


class V2LLM(BaseLLMProvider):
    """A v2 provider: usage only on the response, optional native schema mode."""

    def __init__(
        self, *replies: Any, native: bool = False, images: bool = False
    ) -> None:
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls: list[tuple[Message, ...]] = []
        self.schemas: list[dict[str, Any]] = []
        self.capabilities = LLMCapabilities(
            chat=True,
            structured_output=StructuredOutputMode.SCHEMA
            if native
            else StructuredOutputMode.NONE,
            content_types=frozenset({"text", "image"})
            if images
            else frozenset({"text"}),
        )

    @property
    def name(self) -> str:
        return "v2:model"

    def chat(self, messages, *, options=None) -> LLMResponse:
        self.calls.append(tuple(messages))
        return LLMResponse(
            text=self.replies.pop(0), usage=TokenUsage(5, 1), model=self.name
        )

    def _structured_complete(self, messages, request, options) -> LLMResponse:
        self.schemas.append(request.schema)
        return self.chat(messages, options=options)


def test_native_structured_output_is_used_and_reported():
    llm = V2LLM(answer(("billing", 0.9)), native=True)
    result = Classifier(LABELS, llm=llm).classify("refund please")

    assert result.label == "billing"
    assert result.metadata["structured_output"] == "native"
    enum = llm.schemas[0]["properties"]["labels"]["items"]["properties"]["label"][
        "enum"
    ]
    assert enum == ["billing", "technical", "account"]

    llm = V2LLM(answer(("billing", 0.9)), native=True)
    off = Classifier(LABELS, llm=llm, native_structured_output=False).classify("refund")
    assert off.metadata["structured_output"] == "prompt" and llm.schemas == []


def test_usage_is_read_from_the_response_not_last_usage():
    llm = V2LLM(answer(("billing", 0.9)))
    result = Classifier(LABELS, llm=llm, enable_token_tracking=False).classify("refund")
    assert result.usage == TokenUsage(5, 1)


def test_prompts_come_from_the_registry_and_are_reported():
    registry = PromptRegistry(
        [PromptTemplate("classification.single_label", "2", "Pick exactly one label.")]
    )
    llm = ScriptedLLM(answer(("billing", 0.9)))
    result = Classifier(LABELS, llm=llm, prompts=registry).classify("refund")

    system, _ = llm.calls[0]
    assert "Pick exactly one label." in system
    assert result.metadata["prompts"]["mode"]["version"] == "2"
    assert result.metadata["prompts"]["user"]["name"] == "classification.user"


def test_a_bad_prompt_override_fails_at_construction():
    registry = PromptRegistry(
        [PromptTemplate("classification.user", "2", "{{text}} only")]
    )
    with pytest.raises(ConfigError):
        Classifier(LABELS, llm=ScriptedLLM(), prompts=registry)


def test_repair_calls_report_the_repair_prompt():
    llm = ScriptedLLM("garbage", answer(("billing", 0.9)))
    result = Classifier(LABELS, llm=llm).classify("refund")
    assert result.metadata["prompts"]["user"]["name"] == "classification.repair"


def test_session_supplies_history_and_records_the_turn():
    session = InMemorySessionStore().create(AuthContext("alice", is_authenticated=True))
    session.append(Message.user("my card was charged"))
    llm = ScriptedLLM(answer(("billing", 0.9)))
    result = Classifier(LABELS, llm=llm).classify("yes, refund it", session=session)

    _, prompt = llm.calls[0]
    assert "user: my card was charged" in prompt
    assert session.messages[-1].text == "yes, refund it"
    assert session.state[SESSION_NAMESPACE]["labels"] == ["billing"]
    assert session.state[SESSION_NAMESPACE]["trace_id"] == result.trace_id

    with pytest.raises(ConfigError):
        Classifier(LABELS, llm=ScriptedLLM()).classify(
            "x", session=session, history=[Message.user("y")]
        )


def test_attachments_are_sent_to_a_capable_provider():
    photo = ImagePart.from_bytes(tiny_png())
    llm = V2LLM(answer(("technical", 0.9)), images=True)
    result = Classifier(LABELS, llm=llm).classify(attachments=[photo])

    assert result.label == "technical"
    user = llm.calls[0][-1]
    assert user.parts[0] == photo
    with pytest.raises(UnsupportedContentError):
        Classifier(LABELS, llm=ScriptedLLM()).classify("x", attachments=[photo])


def test_history_renders_multipart_messages_as_text():
    llm = ScriptedLLM(answer(("billing", 0.9)))
    history = [Message.user([TextPart("charged "), TextPart("twice")])]
    Classifier(LABELS, llm=llm).classify("refund", history=history)
    _, prompt = llm.calls[0]
    assert "user: charged twice" in prompt and "TextPart" not in prompt


@dataclass
class _Page:
    number: int
    text: str


@dataclass
class _Doc:
    source: str
    pages: tuple[_Page, ...]


def test_a_paged_document_is_classified():
    llm = ScriptedLLM(answer(("billing", 0.9)))
    doc = _Doc("invoice.pdf", (_Page(1, "Invoice 42"), _Page(2, "Total due")))
    result = Classifier(LABELS, llm=llm).classify(doc)

    assert result.status is ClassificationStatus.DECIDED
    assert result.metadata["document"] == {"source": "invoice.pdf", "pages": [1, 2]}
    _, prompt = llm.calls[0]
    assert "Invoice 42\n\nTotal due" in prompt
