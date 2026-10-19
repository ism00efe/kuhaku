"""Regression tests for the LLM-provider review findings (fix/llm-providers).

One section per finding; every test is offline (``requests.post`` and the google-genai
client are faked).
"""

from __future__ import annotations

import json
import threading
import time

import pytest
import requests

from kuhaku.core.llm import LLMError
from kuhaku.core.llm.anthropic_provider import AnthropicProvider
from kuhaku.core.llm.base import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from kuhaku.core.llm.limits import is_rate_limit_error
from kuhaku.core.llm.openai_provider import OpenAIProvider
from kuhaku.core.retry import CircuitBreaker, CircuitOpenError, CircuitState

_FAST = {"retry_backoff_base_seconds": 0.001, "retry_backoff_max_seconds": 0.01}


class FakeResponse:
    def __init__(self, payload=None, *, status=200, headers=None, lines=None):
        self._payload = payload
        self.status_code = status
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self._lines = lines or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._payload

    def iter_lines(self):
        yield from self._lines

    def close(self):
        pass


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def payload(self) -> dict:
        return self.calls[-1]["json"]


def _sse(*events) -> list[bytes]:
    lines: list[bytes] = []
    for event in events:
        lines += [f"data: {json.dumps(event)}".encode(), b""]
    return lines


def _anthropic_text(text="ok"):
    return FakeResponse({"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"})


# --- typed errors ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, LLMBadRequestError),
        (401, LLMAuthError),
        (403, LLMAuthError),
        (404, LLMBadRequestError),
        (408, LLMTimeoutError),
        (413, LLMBadRequestError),
        (429, LLMRateLimitError),
        (500, LLMServerError),
        (503, LLMServerError),
    ],
)
def test_http_status_maps_to_a_typed_llm_error(monkeypatch, status, error_type):
    monkeypatch.setattr(requests, "post", Recorder(FakeResponse(status=status)))
    provider = OpenAIProvider(api_key="k", model="m", retry_enabled=False)
    with pytest.raises(error_type) as info:
        provider.generate("s", "u")
    assert isinstance(info.value, LLMError)
    assert info.value.status_code == status


def test_transport_failures_map_to_typed_errors(monkeypatch):
    monkeypatch.setattr(requests, "post", Recorder(requests.ConnectTimeout("slow")))
    with pytest.raises(LLMTimeoutError):
        AnthropicProvider(api_key="k", model="m", retry_enabled=False).generate("s", "u")
    monkeypatch.setattr(requests, "post", Recorder(requests.ConnectionError("refused")))
    with pytest.raises(LLMConnectionError):
        AnthropicProvider(api_key="k", model="m", retry_enabled=False).generate("s", "u")


def test_rate_limit_error_carries_retry_after_and_is_recognised(monkeypatch):
    response = FakeResponse(status=429, headers={"Retry-After": "120"})
    monkeypatch.setattr(requests, "post", Recorder(response))
    with pytest.raises(LLMRateLimitError) as info:
        OpenAIProvider(api_key="k", model="m", **_FAST).generate("s", "u")
    assert info.value.retry_after == 120
    assert is_rate_limit_error(info.value)


# --- finding 5: 4xx does not open the breaker -----------------------------------------------


def test_bad_requests_do_not_open_the_circuit_breaker(monkeypatch):
    post = Recorder(FakeResponse(status=400))
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(
        api_key="k", model="m", retry_enabled=False, circuit_breaker_failure_threshold=2
    )
    for _ in range(5):
        with pytest.raises(LLMBadRequestError):
            provider.generate("s", "u")
    assert provider._circuit_breaker.state is CircuitState.CLOSED
    assert len(post.calls) == 5  # every call reached the API; none was fast-failed


def test_server_errors_still_open_the_circuit_breaker(monkeypatch):
    post = Recorder(FakeResponse(status=503))
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(
        api_key="k", model="m", retry_enabled=False, circuit_breaker_failure_threshold=2
    )
    for _ in range(2):
        with pytest.raises(LLMServerError):
            provider.generate("s", "u")
    assert provider._circuit_breaker.state is CircuitState.OPEN


# --- finding 6: Retry-After is honoured; 429 never opens the breaker -----------------------


def test_provider_retry_waits_for_retry_after(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    post = Recorder(FakeResponse(status=429, headers={"Retry-After": "30"}), _anthropic_text())
    monkeypatch.setattr(requests, "post", post)
    provider = AnthropicProvider(api_key="k", model="m", retry_max_attempts=3, **_FAST)
    assert provider.generate("s", "u") == "ok"
    assert sleeps == [30.0]


def test_retry_after_beyond_the_cap_is_not_slept_through(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    post = Recorder(FakeResponse(status=429, headers={"Retry-After": "3600"}))
    monkeypatch.setattr(requests, "post", post)
    provider = AnthropicProvider(api_key="k", model="m", retry_max_attempts=3, **_FAST)
    with pytest.raises(LLMRateLimitError) as info:
        provider.generate("s", "u")
    assert info.value.retry_after == 3600
    assert sleeps == [] and len(post.calls) == 1


def test_exhausted_rate_limits_do_not_open_the_breaker(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    post = Recorder(FakeResponse(status=429, headers={"Retry-After": "1"}))
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(
        api_key="k", model="m", retry_max_attempts=2, circuit_breaker_failure_threshold=1, **_FAST
    )
    for _ in range(3):
        with pytest.raises(LLMRateLimitError) as info:
            provider.generate("s", "u")
        # The 429 stays visible to a RateLimitedLLM -- no CircuitOpenError in its place.
        assert is_rate_limit_error(info.value)
    assert provider._circuit_breaker.state is CircuitState.CLOSED


# --- finding 19: half-open admits a single probe -----------------------------------------


def test_half_open_breaker_admits_one_probe_at_a_time():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.0)
    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert breaker.state is CircuitState.OPEN

    entered = threading.Event()
    release = threading.Event()
    results: list[str] = []

    def probe():
        entered.set()
        release.wait(5)
        return "probe"

    thread = threading.Thread(target=lambda: results.append(breaker.call(probe)))
    thread.start()
    assert entered.wait(5)
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "second caller")
    release.set()
    thread.join(5)
    assert results == ["probe"]
    assert breaker.state is CircuitState.CLOSED
    assert breaker.call(lambda: "after") == "after"


def test_a_neutral_probe_outcome_frees_the_probe_slot():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.0)
    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))

    def bad_request():
        raise ValueError("400")

    with pytest.raises(ValueError):
        breaker.call(bad_request, is_failure=lambda exc: False)
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.call(lambda: "next probe") == "next probe"
    assert breaker.state is CircuitState.CLOSED


# --- finding 1: Anthropic temperature / max_tokens --------------------------------------


def test_current_claude_models_get_no_temperature_even_when_configured(monkeypatch, caplog):
    post = Recorder(_anthropic_text())
    monkeypatch.setattr(requests, "post", post)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5", temperature=0.3)
    provider.generate("s", "u")
    provider.generate("s", "u")
    assert "temperature" not in post.payload
    assert post.payload["max_tokens"] == 16000  # room for adaptive thinking
    warnings = [r for r in caplog.records if "does not accept temperature" in r.getMessage()]
    assert len(warnings) == 1  # warned once, not per call


def test_legacy_claude_models_keep_an_explicit_temperature(monkeypatch):
    post = Recorder(_anthropic_text())
    monkeypatch.setattr(requests, "post", post)
    AnthropicProvider(api_key="k", model="claude-sonnet-4-6", temperature=0.3).generate("s", "u")
    assert post.payload["temperature"] == 0.3 and post.payload["max_tokens"] == 1024
    AnthropicProvider(api_key="k", model="claude-sonnet-4-6").generate("s", "u")
    assert "temperature" not in post.payload


@pytest.mark.parametrize(
    ("model", "sampling", "forced"),
    [
        ("claude-3-5-haiku-20241022", True, True),
        ("claude-opus-4-1-20250805", True, True),
        ("claude-sonnet-4-20250514", True, True),
        ("claude-opus-4-6", True, True),
        ("claude-haiku-4-5", True, True),
        ("claude-opus-4-7", False, True),
        ("claude-opus-4-8", False, True),
        ("claude-sonnet-5", False, True),
        ("claude-opus-5", False, True),
        ("claude-opus-5-5", False, False),
        ("claude-sonnet-5-5", False, False),
        ("claude-fable-5-1", False, False),
        ("my-proxy-alias", True, True),
    ],
)
def test_model_traits(model, sampling, forced):
    from kuhaku.core.llm.anthropic_provider import model_traits

    traits = model_traits(model)
    assert traits.sampling is sampling
    assert traits.forced_tool_choice is forced


def test_build_anthropic_leaves_unchanged_defaults_to_the_provider():
    from kuhaku.core.config import Settings
    from kuhaku.core.resolve.adapters.llm import build_anthropic

    provider = build_anthropic(Settings(anthropic_api_key="k"))
    assert provider._temperature is None and provider._max_tokens == 16000
    provider = build_anthropic(
        Settings(anthropic_api_key="k", llm_temperature=0.0, llm_max_tokens=500)
    )
    assert provider._temperature == 0.0 and provider._max_tokens == 500


def test_default_anthropic_model_is_current():
    from kuhaku.core.config import Settings

    assert Settings().anthropic_model == "claude-opus-5-5"


# --- finding 2: native structured outputs instead of forced tools -------------------------


_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string", "maxLength": 20}},
    "required": ["name"],
}


def test_current_claude_structured_output_uses_output_config(monkeypatch):
    from kuhaku.core.models import Message

    post = Recorder(_anthropic_text('{"name": "Ada"}'))
    monkeypatch.setattr(requests, "post", post)
    result = AnthropicProvider(api_key="k", model="claude-opus-5-5").generate_structured(
        [Message.user("who?")], _SCHEMA, name="person"
    )
    assert result.data == {"name": "Ada"} and result.native
    assert "tools" not in post.payload and "tool_choice" not in post.payload
    assert post.payload["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        }
    }
    assert _SCHEMA["properties"]["name"]["maxLength"] == 20  # caller's schema untouched


def test_structured_refusal_raises_a_refusal_error(monkeypatch):
    from kuhaku.core.llm.base import LLMRefusalError
    from kuhaku.core.models import Message

    reply = FakeResponse({"content": [], "stop_reason": "refusal"})
    monkeypatch.setattr(requests, "post", Recorder(reply))
    with pytest.raises(LLMRefusalError):
        AnthropicProvider(api_key="k", model="claude-opus-5-5").generate_structured(
            [Message.user("q")], _SCHEMA
        )


def test_forced_tool_choice_is_steered_not_forced_on_models_that_reject_it(monkeypatch):
    from kuhaku.core.llm import ToolSpec
    from kuhaku.core.models import Message

    post = Recorder(_anthropic_text())
    monkeypatch.setattr(requests, "post", post)
    tools = [ToolSpec("get_time"), ToolSpec("get_weather")]
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    provider.chat_with_tools(
        [Message.system("sys"), Message.user("q")], tools, tool_choice="get_weather"
    )
    assert post.payload["tool_choice"] == {"type": "auto"}
    assert post.payload["system"].startswith("sys\n\n")
    assert "`get_weather`" in post.payload["system"]
    provider.chat_with_tools([Message.user("q")], tools, tool_choice="required")
    assert post.payload["tool_choice"] == {"type": "auto"}
    assert "one of the provided tools" in post.payload["system"]
    # Where forced tool use is valid it is still sent.
    AnthropicProvider(api_key="k", model="claude-opus-5").chat_with_tools(
        [Message.user("q")], tools, tool_choice="required"
    )
    assert post.payload["tool_choice"] == {"type": "any"} and "system" not in post.payload


# --- finding 13: non-object root schemas keep working $refs --------------------------------


def test_wrapped_schema_hoists_defs_to_the_root(monkeypatch):
    from kuhaku.core.models import Message

    schema = {
        "type": "array",
        "items": {"$ref": "#/$defs/Item"},
        "$defs": {"Item": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    }
    reply = {"content": [{"type": "tool_use", "name": "response", "input": {"value": [{"n": 1}]}}]}
    post = Recorder(FakeResponse(reply))
    monkeypatch.setattr(requests, "post", post)
    result = AnthropicProvider(api_key="k", model="m").generate_structured(
        [Message.user("q")], schema
    )
    assert result.data == [{"n": 1}]
    sent = post.payload["tools"][0]["input_schema"]
    assert sent["$defs"] == schema["$defs"]
    assert "$defs" not in sent["properties"]["value"]
    assert sent["properties"]["value"]["items"] == {"$ref": "#/$defs/Item"}

    post = Recorder(_anthropic_text('{"value": [{"n": 2}]}'))
    monkeypatch.setattr(requests, "post", post)
    result = AnthropicProvider(api_key="k", model="claude-opus-5-5").generate_structured(
        [Message.user("q")], schema
    )
    assert result.data == [{"n": 2}]
    sent = post.payload["output_config"]["format"]["schema"]
    assert "Item" in sent["$defs"] and "$defs" not in sent["properties"]["value"]


# --- finding 3: thinking blocks survive a tool loop ---------------------------------------


_THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}


def test_thinking_blocks_are_replayed_verbatim_in_a_tool_loop(monkeypatch):
    from kuhaku.core.llm import ToolSpec
    from kuhaku.core.models import Message

    first = FakeResponse(
        {
            "content": [
                _THINKING,
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "tu_1", "name": "get_time", "input": {}},
            ],
            "stop_reason": "tool_use",
        }
    )
    post = Recorder(first, _anthropic_text("noon"))
    monkeypatch.setattr(requests, "post", post)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    tools = [ToolSpec("get_time")]
    history = [Message.user("time?")]
    response = provider.chat_with_tools(history, tools)
    assert response.tool_calls[0].id == "tu_1"
    history += [response.message, Message.tool_result("tu_1", "12:00")]
    provider.chat_with_tools(history, tools)
    assistant = post.payload["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == first.json()["content"]  # signature included


def test_replies_without_thinking_carry_no_provider_state(monkeypatch):
    from kuhaku.core.models import Message

    monkeypatch.setattr(requests, "post", Recorder(_anthropic_text()))
    response = AnthropicProvider(api_key="k", model="claude-opus-5-5").chat([Message.user("q")])
    assert response.provider_state is None and response.message == Message.assistant("ok")


# --- finding 18: a failed stream stays failed --------------------------------------------


# --- finding 4: OpenAI reasoning models --------------------------------------------------


def _openai(content="ok", finish="stop", **message):
    return FakeResponse(
        {"choices": [{"message": {"content": content, **message}, "finish_reason": finish}]}
    )


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5.1", "o3", "o4-mini", "o1"])
def test_reasoning_models_get_max_completion_tokens_and_no_temperature(monkeypatch, model):
    post = Recorder(_openai())
    monkeypatch.setattr(requests, "post", post)
    OpenAIProvider(api_key="k", model=model, temperature=0.3, max_tokens=900).generate("s", "u")
    assert post.payload["max_completion_tokens"] == 900
    assert "max_tokens" not in post.payload and "temperature" not in post.payload


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1", "openai/gpt-oss-120b", "o-custom"])
def test_other_models_keep_max_tokens_and_temperature(monkeypatch, model):
    post = Recorder(_openai())
    monkeypatch.setattr(requests, "post", post)
    OpenAIProvider(api_key="k", model=model).generate("s", "u")
    assert post.payload["max_tokens"] == 1024 and post.payload["temperature"] == 0.1


# --- finding 20: null content without tool calls -----------------------------------------


def test_refusal_raises_a_refusal_error_without_dumping_the_body(monkeypatch):
    from kuhaku.core.llm.base import LLMRefusalError

    monkeypatch.setattr(requests, "post", Recorder(_openai(None, refusal="I can't help.")))
    with pytest.raises(LLMRefusalError, match="I can't help") as info:
        OpenAIProvider(api_key="k", model="gpt-5").generate("s", "u")
    assert "choices" not in str(info.value)


def test_length_cut_reasoning_reply_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(requests, "post", Recorder(_openai(None, finish="length")))
    with pytest.raises(LLMError, match="token limit") as info:
        OpenAIProvider(api_key="k", model="gpt-5").generate("s", "u")
    assert "choices" not in str(info.value)


def test_null_content_with_tool_calls_is_still_fine(monkeypatch):
    from kuhaku.core.llm import ToolSpec
    from kuhaku.core.models import Message

    calls = [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
    monkeypatch.setattr(requests, "post", Recorder(_openai(None, "tool_calls", tool_calls=calls)))
    response = OpenAIProvider(api_key="k", model="m").chat_with_tools(
        [Message.user("q")], [ToolSpec("t")]
    )
    assert response.text == "" and response.tool_calls[0].name == "t"


# --- finding 16: Groq defaults, structured output, vision -------------------------------


def test_groq_default_model_and_json_schema_structured_output(monkeypatch):
    from kuhaku.core.config import Settings
    from kuhaku.core.llm import StructuredOutputMode
    from kuhaku.core.llm.groq_provider import GroqProvider
    from kuhaku.core.models import Message

    assert Settings().groq_model == "openai/gpt-oss-20b"
    post = Recorder(_openai('{"name": "Ada"}'))
    monkeypatch.setattr(requests, "post", post)
    provider = GroqProvider(api_key="k", model="openai/gpt-oss-120b")
    assert provider.capabilities.structured_output is StructuredOutputMode.SCHEMA
    result = provider.generate_structured([Message.user("who?")], _SCHEMA, name="person")
    assert result.data == {"name": "Ada"}
    assert post.payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "person", "schema": _SCHEMA, "strict": False},
    }
    # Models without json_schema support keep JSON mode.
    other = GroqProvider(api_key="k", model="llama-3.1-8b-instant")
    assert other.capabilities.structured_output is StructuredOutputMode.JSON


def test_groq_vision_follows_current_models():
    from kuhaku.core.llm.groq_provider import GroqProvider

    assert "image" in GroqProvider(api_key="k", model="qwen/qwen3.8-27b").capabilities.content_types
    retired = GroqProvider(api_key="k", model="meta-llama/llama-4-scout-17b-16e-instruct")
    assert retired.capabilities.content_types == frozenset({"text"})


# --- finding 17: RAG token metrics are labelled with the real provider ------------------


def test_rag_token_tracking_uses_the_built_providers_label(monkeypatch):
    import kuhaku
    from kuhaku.core.config import Settings
    from tests.conftest import FakeEmbeddings, FakeLLM, FakeVectorStore

    class NamedLLM(FakeLLM):
        @property
        def name(self) -> str:
            return "groq:openai/gpt-oss-120b"

    monkeypatch.setattr(kuhaku, "build_embedding_provider", lambda rs: FakeEmbeddings())
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: FakeVectorStore())
    monkeypatch.setattr(
        "kuhaku.tools.rag.resolve.adapters.embedding._model_cached", lambda name: True
    )
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s, **_k: NamedLLM())
    rag = kuhaku.RAG(settings=Settings(_env_file=None, audit_enabled=False), cache=False)
    assert rag._llm._provider == "groq"  # not the configured "auto"


# --- findings 7, 8, 9: Vertex AI ---------------------------------------------------------


@pytest.fixture
def vertex(monkeypatch):
    pytest.importorskip("google.genai")
    from google import genai

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    class Models:
        def __init__(self, replies):
            self.replies = list(replies)
            self.calls: list[dict] = []
            self.stream_calls: list[dict] = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return self.replies.pop(0)

        def generate_content_stream(self, **kwargs):
            self.stream_calls.append(kwargs)
            yield from self.replies.pop(0)

    def make(replies, **kwargs):
        models = Models(replies)
        monkeypatch.setattr(genai, "Client", lambda **_: type("C", (), {"models": models})())
        return VertexAIProvider(project="p", location="l", **kwargs), models

    return make


def _gemini(parts, *, prompt=3, candidates=4, thoughts=None):
    from google.genai import types

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts), finish_reason="STOP"
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            thoughts_token_count=thoughts,
        ),
    )


def _signed_call():
    from google.genai import types

    return types.Part(
        function_call=types.FunctionCall(name="get_time", args={"tz": "UTC"}),
        thought_signature=b"SIG",
    )


def test_vertex_thought_signature_round_trips(vertex):
    from google.genai import types

    from kuhaku.core.llm import ToolSpec
    from kuhaku.core.models import Message

    provider, models = vertex([_gemini([_signed_call()]), _gemini([types.Part(text="noon")])])
    tools = [ToolSpec("get_time")]
    response = provider.chat_with_tools([Message.user("t?")], tools)
    call = response.tool_calls[0]
    history = [Message.user("t?"), response.message, Message.tool_result(call.id, "12:00")]
    assert provider.chat_with_tools(history, tools).text == "noon"
    model_turn = models.calls[1]["contents"][1]
    assert model_turn == {
        "role": "model",
        "parts": [
            {"function_call": {"name": "get_time", "args": {"tz": "UTC"}},
             "thought_signature": b"SIG"}
        ],
    }


def test_vertex_usage_counts_thinking_tokens(vertex):
    from google.genai import types

    provider, _ = vertex([_gemini([types.Part(text="hi")], candidates=4, thoughts=10)])
    from kuhaku.core.models import Message

    response = provider.chat([Message.user("q")])
    assert response.usage.completion_tokens == 14
    assert response.provider_state is None  # nothing signed, nothing to carry


def test_vertex_constructor_defaults_are_sent_and_overridable(vertex):
    from google.genai import types

    from kuhaku.core.llm import GenerationOptions
    from kuhaku.core.models import Message

    replies = [_gemini([types.Part(text="a")]) for _ in range(3)]
    provider, models = vertex(replies, temperature=0.2, max_tokens=300)
    provider.chat([Message.user("q")])
    assert models.calls[0]["config"]["temperature"] == 0.2
    assert models.calls[0]["config"]["max_output_tokens"] == 300
    provider.chat([Message.user("q")], options=GenerationOptions(temperature=0.9))
    assert models.calls[1]["config"]["temperature"] == 0.9
    bare, models = vertex(replies[2:])
    bare.chat([Message.user("q")])
    assert "temperature" not in models.calls[0]["config"]


def test_build_vertex_wires_configured_llm_settings(vertex, monkeypatch):
    from google import genai

    from kuhaku.core.config import Settings
    from kuhaku.core.resolve.adapters.llm import build_vertex

    _, models = vertex([])
    monkeypatch.setattr(genai, "Client", lambda **_: type("C", (), {"models": models})())
    settings = Settings(vertex_project="p", vertex_location="l", llm_temperature=0.4,
                        llm_max_tokens=777)
    provider = build_vertex(settings)
    assert provider._temperature == 0.4 and provider._max_tokens == 777
    assert provider.name == "vertex:gemini-3.5-flash"
    untouched = build_vertex(Settings(vertex_project="p", vertex_location="l"))
    assert untouched._temperature is None and untouched._max_tokens is None
