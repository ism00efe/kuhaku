"""Tests for the sentence-transformers embedding wrapper (E5 prefixing).

A fake SentenceTransformer is injected so no real model is downloaded/loaded.
"""

from __future__ import annotations

import numpy as np
import pytest
import sentence_transformers

from kuhaku.tools.rag.embeddings import SentenceTransformerEmbeddings


class FakeST:
    last_texts: list[str] = []

    def __init__(self, model_name, device=None):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True,
               show_progress_bar=False):
        FakeST.last_texts = list(texts)
        return np.array([[1.0, 2.0, 3.0] for _ in texts])


@pytest.fixture(autouse=True)
def _patch_st(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    FakeST.last_texts = []


def test_e5_model_adds_passage_and_query_prefixes():
    emb = SentenceTransformerEmbeddings("intfloat/multilingual-e5-small")

    emb.embed_documents(["hello"])
    assert FakeST.last_texts == ["passage: hello"]

    emb.embed_query("world")
    assert FakeST.last_texts == ["query: world"]


def test_non_e5_model_no_prefixes():
    emb = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    emb.embed_documents(["hello"])
    assert FakeST.last_texts == ["hello"]
    emb.embed_query("world")
    assert FakeST.last_texts == ["world"]


def test_returns_plain_lists():
    emb = SentenceTransformerEmbeddings("intfloat/multilingual-e5-small")
    docs = emb.embed_documents(["a", "b"])
    assert isinstance(docs, list) and isinstance(docs[0], list)
    assert emb.embed_query("q") == [1.0, 2.0, 3.0]


# --- Retry (D40): embed_query only, not embed_documents/ingestion -----------------
class FlakyFakeST:
    """A SentenceTransformer stand-in whose .encode() fails `fail_times` times, then
    succeeds -- used to exercise embed_query's retry wrapping."""

    fail_times: int = 0
    calls: int = 0

    def __init__(self, model_name, device=None):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True,
               show_progress_bar=False):
        FlakyFakeST.calls += 1
        if FlakyFakeST.calls <= FlakyFakeST.fail_times:
            raise RuntimeError(f"transient failure #{FlakyFakeST.calls}")
        return np.array([[1.0, 2.0, 3.0] for _ in texts])


_FAST_RETRY = dict(retry_backoff_base_seconds=0.001, retry_backoff_max_seconds=0.01)


def test_embed_query_retries_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FlakyFakeST)
    FlakyFakeST.fail_times, FlakyFakeST.calls = 1, 0

    emb = SentenceTransformerEmbeddings("intfloat/multilingual-e5-small", **_FAST_RETRY)
    assert emb.embed_query("world") == [1.0, 2.0, 3.0]
    assert FlakyFakeST.calls == 2


def test_embed_query_exhausts_retries_and_raises_original_exception(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FlakyFakeST)
    FlakyFakeST.fail_times, FlakyFakeST.calls = 99, 0

    emb = SentenceTransformerEmbeddings(
        "intfloat/multilingual-e5-small", retry_max_attempts=2, **_FAST_RETRY
    )
    with pytest.raises(RuntimeError, match="transient failure #2"):
        emb.embed_query("world")
    assert FlakyFakeST.calls == 2


def test_embed_query_retry_disabled_bypasses_retry(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FlakyFakeST)
    FlakyFakeST.fail_times, FlakyFakeST.calls = 99, 0

    emb = SentenceTransformerEmbeddings(
        "intfloat/multilingual-e5-small", retry_enabled=False
    )
    with pytest.raises(RuntimeError, match="transient failure #1"):
        emb.embed_query("world")
    assert FlakyFakeST.calls == 1


def test_embed_documents_is_not_retried_ingestion_out_of_scope(monkeypatch):
    """embed_documents is the ingestion-path method -- retry is explicitly out of scope
    for ingestion (D40), so a failure there must propagate on the first call."""

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FlakyFakeST)
    FlakyFakeST.fail_times, FlakyFakeST.calls = 99, 0

    emb = SentenceTransformerEmbeddings("intfloat/multilingual-e5-small", **_FAST_RETRY)
    with pytest.raises(RuntimeError, match="transient failure #1"):
        emb.embed_documents(["a"])
    assert FlakyFakeST.calls == 1
