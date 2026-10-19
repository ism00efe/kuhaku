"""Image and document parts on the wire: per-provider encoding, model-aware
``content_types``, and rejection of what a provider/model cannot take.

No network -- ``requests.post`` and ``google.genai.Client`` are monkeypatched.
"""

from __future__ import annotations

import logging

import pytest
import requests

from kuhaku import DocumentPart, ImagePart, Message, Settings, TextPart
from kuhaku.core.llm import LegacyLLMAdapter, LLMError, UnsupportedContentError
from kuhaku.core.llm.anthropic_provider import AnthropicProvider
from kuhaku.core.llm.groq_provider import GroqProvider
from kuhaku.core.llm.ollama_provider import OllamaProvider
from kuhaku.core.llm.openai_provider import OpenAIProvider
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.resolve.adapters.llm import build_anthropic, build_ollama, build_openai
from tests.test_media import PDF, tiny_png

ALL = frozenset({"text", "image", "document"})
TEXT = frozenset({"text"})


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class Recorder:
    def __init__(self, payload):
        self.payload_to_return = payload
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return FakeResponse(self.payload_to_return)

    @property
    def payload(self) -> dict:
        return self.calls[-1]["json"]


OPENAI_REPLY = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
ANTHROPIC_REPLY = {"content": [{"type": "text", "text": "ok"}]}
OLLAMA_REPLY = {"message": {"content": "ok"}}


@pytest.fixture
def image() -> ImagePart:
    return ImagePart.from_bytes(tiny_png(), detail="high")


@pytest.fixture
def pdf() -> DocumentPart:
    return DocumentPart.from_bytes(PDF, filename="invoice.pdf")


# --- capabilities ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (lambda: OpenAIProvider(api_key="k", model="gpt-4o-mini"), ALL),
        (lambda: OpenAIProvider(api_key="k", model="gpt-4.1-2025-04-14"), ALL),
        (lambda: OpenAIProvider(api_key="k", model="gpt-4o-audio-preview"), TEXT),
        (lambda: OpenAIProvider(api_key="k", model="o3-mini"), TEXT),
        (lambda: OpenAIProvider(api_key="k", model="gpt-3.5-turbo"), TEXT),
        # Another server behind the OpenAI shape: the model is unknown.
        (lambda: OpenAIProvider(api_key="k", model="gpt-4o", base_url="http://vllm:8000/v1"), TEXT),
        (lambda: GroqProvider(api_key="k", model="llama-3.3-70b-versatile"), TEXT),
        (
            lambda: GroqProvider(api_key="k", model="qwen/qwen3.8-27b"),
            frozenset({"text", "image"}),
        ),
        (lambda: AnthropicProvider(api_key="k", model="claude-sonnet-5"), ALL),
        (lambda: AnthropicProvider(api_key="k", model="my-proxy-model"), TEXT),
        (lambda: OllamaProvider("http://x", "qwen2.5:7b-instruct"), TEXT),
        (lambda: OllamaProvider("http://x", "llava:13b"), frozenset({"text", "image"})),
        (lambda: OllamaProvider("http://x", "library/qwen2.5vl"), frozenset({"text", "image"})),
        (lambda: OllamaProvider("http://x", "gemma3:1b"), TEXT),
    ],
)
def test_default_content_types_follow_the_model(provider, expected):
    assert provider().capabilities.content_types == expected


def test_content_types_can_be_configured():
    provider = OllamaProvider("http://x", "my-finetune", content_types=["image"])
    assert provider.capabilities.content_types == frozenset({"text", "image"})
    assert OpenAIProvider(api_key="k", model="gpt-4o", content_types="text").capabilities.content_types == TEXT
    # A type the provider has no encoding for is refused at construction.
    with pytest.raises(LLMError, match="document"):
        OllamaProvider("http://x", "llava", content_types="text,document")
    with pytest.raises(LLMError, match="document"):
        GroqProvider(api_key="k", model="m", content_types="image,document")
    # The class-level default stays text-only, and other capabilities are kept.
    assert OpenAIProvider.capabilities.content_types == TEXT
    off = OpenAIProvider(api_key="k", model="gpt-4o", native_structured_output=False)
    assert off.capabilities.content_types == ALL and not off.capabilities.native_structured_output


def test_settings_content_types_reach_the_built_provider():
    settings = Settings(
        llm_content_types="text,image",
        openai_api_key="k",
        anthropic_api_key="k",
        ollama_model="my-vision-finetune",
    )
    assert build_ollama(settings).capabilities.content_types == frozenset({"text", "image"})
    assert build_openai(settings).capabilities.content_types == frozenset({"text", "image"})
    assert build_anthropic(settings).capabilities.content_types == frozenset({"text", "image"})
    assert build_ollama(Settings(ollama_model="my-vision-finetune")).capabilities.content_types == TEXT


def test_wrappers_report_the_wrapped_content_types():
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-5")
    assert TokenTrackingLLM(provider, provider="anthropic").capabilities.content_types == ALL


# --- encoding ------------------------------------------------------------------------


def test_openai_encodes_images_and_pdfs(monkeypatch, image, pdf):
    post = Recorder(OPENAI_REPLY)
    monkeypatch.setattr(requests, "post", post)
    OpenAIProvider(api_key="k", model="gpt-4o-mini").chat(
        [Message.system("s"), Message.user([image, pdf, TextPart("read these")])]
    )
    assert post.payload["messages"] == [
        {"role": "system", "content": "s"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image.data_url, "detail": "high"}},
                {"type": "file", "file": {"filename": "invoice.pdf", "file_data": pdf.data_url}},
                {"type": "text", "text": "read these"},
            ],
        },
    ]


def test_openai_sends_image_urls_but_not_document_urls(monkeypatch):
    post = Recorder(OPENAI_REPLY)
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(api_key="k", model="gpt-4o")
    provider.chat([Message.user([ImagePart.from_url("https://example.com/a.png")])])
    assert post.payload["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
    ]
    with pytest.raises(UnsupportedContentError, match="by URL"):
        provider.chat([Message.user([DocumentPart.from_url("https://example.com/a.pdf")])])
    with pytest.raises(UnsupportedContentError, match="gs://"):
        provider.chat([Message.user([ImagePart.from_url("gs://bucket/a.png")])])
    assert len(post.calls) == 1


def test_groq_encodes_images_like_openai(monkeypatch, image, pdf):
    post = Recorder(OPENAI_REPLY)
    monkeypatch.setattr(requests, "post", post)
    groq = GroqProvider(api_key="k", model="qwen/qwen3.8-27b")
    groq.chat([Message.user([image, TextPart("what?")])])
    assert post.payload["messages"][0]["content"][0]["type"] == "image_url"
    with pytest.raises(UnsupportedContentError, match="document"):
        groq.chat([Message.user([pdf])])
    assert len(post.calls) == 1


def test_anthropic_encodes_image_and_document_blocks(monkeypatch, image, pdf):
    post = Recorder(ANTHROPIC_REPLY)
    monkeypatch.setattr(requests, "post", post)
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-5")
    provider.chat(
        [
            Message.user(
                [image, pdf, DocumentPart.from_url("https://example.com/b.pdf"), TextPart("go")]
            )
        ]
    )
    assert post.payload["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": image.base64},
                },
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": pdf.base64},
                    "title": "invoice.pdf",
                },
                {"type": "document", "source": {"type": "url", "url": "https://example.com/b.pdf"}},
                {"type": "text", "text": "go"},
            ],
        }
    ]


def test_ollama_puts_images_in_the_images_field(monkeypatch, image):
    post = Recorder(OLLAMA_REPLY)
    monkeypatch.setattr(requests, "post", post)
    provider = OllamaProvider("http://x", "llava")
    provider.chat([Message.user([TextPart("what is "), image, TextPart("this?")])])
    assert post.payload["messages"] == [
        {"role": "user", "content": "what is this?", "images": [image.base64]}
    ]
    with pytest.raises(UnsupportedContentError, match="URL"):
        provider.chat([Message.user([ImagePart.from_url("https://example.com/a.png")])])
    assert len(post.calls) == 1


def test_text_only_payloads_are_unchanged_on_a_multimodal_model(monkeypatch):
    post = Recorder(OPENAI_REPLY)
    monkeypatch.setattr(requests, "post", post)
    OpenAIProvider(api_key="k", model="gpt-4o").chat(
        [Message.user([TextPart("a"), TextPart("b")])]
    )
    assert post.payload["messages"] == [{"role": "user", "content": "ab"}]


# --- rejection -----------------------------------------------------------------------


def test_unsupported_media_is_rejected_before_any_request(monkeypatch, image, pdf):
    post = Recorder(OPENAI_REPLY)
    monkeypatch.setattr(requests, "post", post)
    text_only = OpenAIProvider(api_key="k", model="gpt-3.5-turbo")
    with pytest.raises(UnsupportedContentError, match="content_types="):
        text_only.chat([Message.user([image])])
    with pytest.raises(UnsupportedContentError, match="document"):
        OllamaProvider("http://x", "llava").chat([Message.user([pdf])])
    vision = OpenAIProvider(api_key="k", model="gpt-4o")
    with pytest.raises(UnsupportedContentError, match="user messages"):
        vision.chat([Message.user("q"), Message.assistant([image])])
    with pytest.raises(UnsupportedContentError):
        vision.chat([Message.system([image]), Message.user("q")])
    assert post.calls == []


def test_structured_path_rejects_too(monkeypatch, image):
    post = Recorder(OPENAI_REPLY)
    monkeypatch.setattr(requests, "post", post)
    provider = OpenAIProvider(api_key="k", model="gpt-3.5-turbo")
    with pytest.raises(UnsupportedContentError):
        provider.generate_structured([Message.user([image])], {"type": "object"})
    assert post.calls == []


def test_legacy_adapter_rejects_media_clearly(image):
    class V1:
        name = "v1:model"

        def generate(self, system, user):  # pragma: no cover - must not be reached
            raise AssertionError("called")

    with pytest.raises(UnsupportedContentError, match="only text"):
        LegacyLLMAdapter(V1()).chat([Message.user([image])])


# --- observability -------------------------------------------------------------------


def test_logs_describe_media_without_content(monkeypatch, caplog, image):
    post = Recorder(OPENAI_REPLY)
    monkeypatch.setattr(requests, "post", post)
    with caplog.at_level(logging.DEBUG):
        TokenTrackingLLM(OpenAIProvider(api_key="k", model="gpt-4o"), provider="openai").chat(
            [Message.user([image, TextPart("q")])]
        )
    media_logs = [r for r in caplog.records if r.getMessage() == "llm media content"]
    assert media_logs and media_logs[0].parts == [image.describe()]
    for record in caplog.records:
        rendered = record.getMessage() + repr(record.__dict__)
        assert image.base64 not in rendered


# --- Vertex ----------------------------------------------------------------------------


class _Reply:
    text = "ok"
    usage_metadata = None
    candidates: tuple = ()


class _Models:
    def __init__(self):
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _Reply()


@pytest.fixture
def vertex(monkeypatch):
    pytest.importorskip("google.genai", reason="needs the optional 'vertex' extra")
    from google import genai

    from kuhaku.core.llm.vertex_provider import VertexAIProvider

    models = _Models()
    monkeypatch.setattr(genai, "Client", lambda **_: type("C", (), {"models": models})())

    def make(**kwargs):
        return VertexAIProvider(project="p", location="l", **kwargs), models

    return make


def test_vertex_encodes_inline_bytes_and_file_uris(vertex, image, pdf):
    provider, models = vertex()
    assert provider.capabilities.content_types == ALL
    provider.chat(
        [Message.user([image, pdf, DocumentPart.from_url("gs://b/c.pdf"), TextPart("go")])]
    )
    assert models.calls[0]["contents"] == [
        {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": image.data}},
                {"inline_data": {"mime_type": "application/pdf", "data": pdf.data}},
                {"file_data": {"file_uri": "gs://b/c.pdf", "mime_type": "application/pdf"}},
                {"text": "go"},
            ],
        }
    ]
    provider.generate("s", "u")
    assert models.calls[-1]["contents"] == "u"  # the text-only shorthand is unchanged


def test_vertex_non_gemini_model_is_text_only(vertex, image):
    provider, models = vertex(model="claude-on-vertex")
    assert provider.capabilities.content_types == TEXT
    with pytest.raises(UnsupportedContentError):
        provider.chat([Message.user([image])])
    assert models.calls == []
