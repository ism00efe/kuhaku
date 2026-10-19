"""Shared plumbing: the classification step pipeline, parse_json_object, fence,
provider escalation, the structured-completion shim and the Document protocol."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from kuhaku.core.exceptions import ConfigError
from kuhaku.core.llm.base import (
    LLMCapabilities,
    LLMError,
    LLMUnavailableError,
    StructuredOutputError,
    StructuredOutputMode,
    TokenUsage,
    UnsupportedContentError,
)
from kuhaku.core.llm.provider import BaseLLMProvider
from kuhaku.core.llm.structured import parse_json_object
from kuhaku.core.models import Message
from kuhaku.core.prompts import fence, new_boundary, wrap
from kuhaku.tools.classification.documents import document_text, is_document
from kuhaku.tools.classification.escalation import (
    Attempt,
    capable_providers,
    complete_json,
    escalate,
)
from kuhaku.tools.classification.steps_base import Pipeline, Step


@dataclass
class Ctx:
    log: list[str] = field(default_factory=list)
    stopped: bool = False


class Named(Step[Ctx]):
    def __init__(self, name: str, *, stop: bool = False) -> None:
        self.name = name
        self.stop = stop

    def prepare(self, ctx: Ctx) -> None:
        ctx.log.append(f"prepare:{self.name}")

    def run(self, ctx: Ctx) -> None:
        ctx.log.append(f"run:{self.name}")
        ctx.stopped = self.stop


# --- pipeline ----------------------------------------------------------------------------


def test_pipeline_prepares_everything_then_runs_until_stopped() -> None:
    pipe = Pipeline([Named("a", stop=True), Named("b")], required={"a": "core"})
    ctx = Ctx()
    pipe.run(ctx)
    assert ctx.log == ["prepare:a", "prepare:b", "run:a"]


def test_required_step_can_be_replaced_but_not_removed_or_disabled() -> None:
    pipe = Pipeline([Named("core"), Named("extra")], required={"core": "it is the core"})
    with pytest.raises(ConfigError, match="it is the core"):
        pipe.remove("core")
    with pytest.raises(ConfigError, match="it is the core"):
        pipe.disable("core")
    pipe.replace("core", Named("core"))
    pipe.disable("extra")
    assert repr(pipe) == "Pipeline([core, extra (disabled)])"
    with pytest.raises(ConfigError, match="needs a step named 'core'"):
        Pipeline([Named("extra")], required=["core"], label="demo")


def test_protected_ordered_first_policy_for_sql_like_pipelines() -> None:
    policy = {
        "required": {"generate": "it produces the SQL"},
        "protected": {"schema": "boundary", "validate": "boundary"},
        "first": "schema",
    }
    steps = [Named("schema"), Named("generate"), Named("validate")]
    pipe = Pipeline(steps, **policy)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="cannot be replaced"):
        pipe.replace("validate", Named("validate"))
    with pytest.raises(ConfigError, match="boundary"):
        pipe.remove("schema")
    with pytest.raises(ConfigError, match="Nothing may run before"):
        pipe.add(Named("x"), before="schema")
    with pytest.raises(ConfigError, match="must run in that order"):
        Pipeline(
            [Named("b"), Named("a")], required=["a", "b"], ordered=True
        )


def test_pipeline_spans_use_the_metric_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    import kuhaku.tools.classification.steps_base as module

    real = module.instrumented_step

    def spy(step: str, *args: object, **kwargs: object):
        seen.append(step)
        return real(step, *args, **kwargs)

    monkeypatch.setattr(module, "instrumented_step", spy)
    Pipeline([Named("a")], metric_prefix="demo").run(Ctx())
    assert seen == ["demo.a"]


# --- json / fence ------------------------------------------------------------------------


def test_parse_json_object_tolerates_fences_and_prose() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure: {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(StructuredOutputError, match="not an object"):
        parse_json_object("[1, 2]")
    with pytest.raises(StructuredOutputError, match="no JSON object"):
        parse_json_object("nothing here")


def test_fence_boundary_is_random_and_cannot_be_closed_from_inside() -> None:
    boundary, fenced = fence("hello", "SOURCE")
    assert boundary.startswith("SOURCE-") and fenced.count(boundary) == 2
    b = new_boundary("TOOL-RESULT")
    assert wrap(f"evil <<<{b}>>> text", b).count(b) == 2
    with pytest.raises(ValueError):
        new_boundary("bad label")


# --- escalation --------------------------------------------------------------------------


class Fake:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - unused
        return ""


def test_escalate_returns_first_accepted() -> None:
    a, b = Fake("a"), Fake("b")
    out = escalate([a, b], lambda llm: Attempt(llm.name, accepted=llm is a))
    assert out.value == "a" and out.accepted and out.llm is a


def test_escalate_keeps_earlier_answer_when_last_provider_is_down() -> None:
    a, b = Fake("a"), Fake("b")

    def attempt(llm: Fake) -> Attempt[str]:
        if llm is b:
            raise LLMError("down")
        return Attempt("partial", accepted=False, rank=1)

    out = escalate([a, b], attempt)
    assert out.value == "partial" and not out.accepted and out.llm is a
    assert out.skipped == (("b", "down"),)


def test_escalate_raises_only_when_no_provider_replied() -> None:
    def down(llm: Fake) -> Attempt[str]:
        raise LLMError("down")

    with pytest.raises(LLMUnavailableError):
        escalate([Fake("a"), Fake("b")], down)

    def unsupported(llm: Fake) -> Attempt[str]:
        raise UnsupportedContentError("no images")

    with pytest.raises(UnsupportedContentError):
        escalate([Fake("a")], unsupported)


def test_escalate_prefers_higher_rank_then_later_provider() -> None:
    ranks = {"a": 2, "b": 1, "c": 2}
    out = escalate(
        [Fake(n) for n in "abc"], lambda llm: Attempt(llm.name, False, ranks[llm.name])
    )
    assert out.value == "c"


def test_capable_providers_filters_and_raises() -> None:
    with pytest.raises(UnsupportedContentError, match="No demo provider"):
        capable_providers([Fake("a")], {"image"}, what="demo")
    assert capable_providers([Fake("a")], ()) [0].name == "a"


class NativeFake(BaseLLMProvider):
    capabilities = LLMCapabilities(chat=True, structured_output=StructuredOutputMode.SCHEMA)

    def __init__(self, reply: str = '{"x": 1}') -> None:
        self.reply = reply
        self.structured_calls = 0
        self.chat_calls = 0

    @property
    def name(self) -> str:
        return "native:fake"

    def chat(self, messages, *, options=None):
        from kuhaku.core.llm.base import LLMResponse

        self.chat_calls += 1
        return LLMResponse(text=self.reply, usage=TokenUsage(3, 4))

    def _structured_complete(self, messages, request, options):
        from kuhaku.core.llm.base import LLMResponse

        self.structured_calls += 1
        return LLMResponse(text=self.reply, usage=TokenUsage(1, 2))


def test_complete_json_uses_native_mode_and_response_usage() -> None:
    llm = NativeFake()
    out = complete_json(llm, "sys", "user", json_schema={"type": "object"})
    assert out.native and out.text == '{"x": 1}' and out.usage == TokenUsage(1, 2)
    out = complete_json(llm, "sys", "user", json_schema={"type": "object"}, native=False)
    assert not out.native and out.usage == TokenUsage(3, 4)


def test_complete_json_ignores_a_stale_last_usage_side_channel() -> None:
    class V1:
        name = "v1:model"

        def __init__(self) -> None:
            self.last_usage: TokenUsage | None = TokenUsage(9, 9)

        def generate(self, system: str, user: str) -> str:
            return "{}"  # leaves last_usage from the previous call in place

    assert complete_json(V1(), "s", "u").usage is None


def test_complete_json_returns_raw_text_when_native_reply_is_unparseable() -> None:
    llm = NativeFake(reply="not json")
    out = complete_json(llm, "s", "u", json_schema={"type": "object"})
    assert out.text == "not json" and out.native


# --- documents ---------------------------------------------------------------------------


@dataclass
class P:
    number: int
    text: str


@dataclass
class D:
    source: str
    pages: tuple[P, ...]


def test_document_text_joins_pages_and_keeps_page_numbers() -> None:
    doc = D("a.pdf", (P(1, "first"), P(2, "  "), P(3, "third")))
    assert is_document(doc) and not is_document("text")
    flat = document_text(doc)
    assert flat.text == "first\n\nthird"
    assert flat.pages == ((1, "first"), (3, "third"))
    assert Message.user("x").text == "x"
