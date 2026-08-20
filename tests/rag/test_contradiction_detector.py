"""Tests for rag/contradiction_detector.py's ContradictionDetector (D50)."""

from __future__ import annotations

import json

from kuhaku.tools.rag.models import RetrievedChunk
from kuhaku.tools.rag.contradiction_detector import ContradictionDetector
from tests.conftest import make_chunk


class _ScriptedEmbeddings:
    """Returns pre-assigned, already-normalized vectors keyed by chunk text."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors_by_text[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors_by_text[text]


class _ScriptedLLM:
    name = "scripted"

    def __init__(self, response: str | None = None, *, raises: Exception | None = None) -> None:
        self._response = response
        self._raises = raises
        self.call_count = 0

    def generate(self, system: str, user: str) -> str:
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        assert self._response is not None
        return self._response


def _retrieved(*chunks) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=c, score=1.0) for c in chunks]


def _judge_json(
    *, contradiction: bool, topic: str = "iade süresi", explanation: str = "farklı"
) -> str:
    return json.dumps(
        {"contradiction": contradiction, "topic": topic, "explanation": explanation}
    )


def test_detect_confirms_a_clear_contradiction_between_similar_chunks():
    chunk_a = make_chunk("doc_a", text="İade süresi 14 gündür.")
    chunk_b = make_chunk("doc_b", text="İade süresi 30 gündür.")
    embedder = _ScriptedEmbeddings({chunk_a.text: [1.0, 0.0], chunk_b.text: [1.0, 0.0]})
    judge = _ScriptedLLM(_judge_json(contradiction=True))

    detector = ContradictionDetector(embedder, judge, similarity_threshold=0.75)
    result = detector.detect(_retrieved(chunk_a, chunk_b))

    assert result.has_contradiction is True
    assert result.warning_message is not None
    assert "[S1]" in result.warning_message
    assert "[S2]" in result.warning_message
    assert len(result.contradiction_pairs) == 1
    pair = result.contradiction_pairs[0]
    assert pair.chunk_a_id == chunk_a.id
    assert pair.chunk_b_id == chunk_b.id
    assert pair.topic == "iade süresi"
    assert judge.call_count == 1


def test_detect_returns_false_when_judge_finds_no_contradiction():
    chunk_a = make_chunk("doc_a", text="İade süresi 14 gündür.")
    chunk_b = make_chunk("doc_b", text="İade süresi 14 gün olarak belirlenmiştir.")
    embedder = _ScriptedEmbeddings({chunk_a.text: [1.0, 0.0], chunk_b.text: [1.0, 0.0]})
    judge = _ScriptedLLM(_judge_json(contradiction=False))

    detector = ContradictionDetector(embedder, judge)
    result = detector.detect(_retrieved(chunk_a, chunk_b))

    assert result.has_contradiction is False
    assert result.warning_message is None
    assert result.contradiction_pairs == []


def test_detect_skips_low_similarity_pairs_without_calling_the_judge():
    chunk_a = make_chunk("doc_a", text="İade süresi 14 gündür.")
    chunk_b = make_chunk("doc_b", text="EFT limiti 50000 TL'dir.")
    # Orthogonal vectors -> cosine similarity 0.0, well below the default threshold.
    embedder = _ScriptedEmbeddings({chunk_a.text: [1.0, 0.0], chunk_b.text: [0.0, 1.0]})
    judge = _ScriptedLLM(_judge_json(contradiction=True))

    detector = ContradictionDetector(embedder, judge, similarity_threshold=0.75)
    result = detector.detect(_retrieved(chunk_a, chunk_b))

    assert result.has_contradiction is False
    assert judge.call_count == 0


def test_detect_treats_a_scope_difference_as_not_a_contradiction():
    chunk_a = make_chunk("doc_a", text="Bireysel hesaplar için EFT limiti 50000 TL'dir.")
    chunk_b = make_chunk("doc_b", text="Kurumsal hesaplar için EFT limiti 500000 TL'dir.")
    embedder = _ScriptedEmbeddings({chunk_a.text: [1.0, 0.0], chunk_b.text: [1.0, 0.0]})
    judge = _ScriptedLLM(_judge_json(contradiction=False))

    detector = ContradictionDetector(embedder, judge)
    result = detector.detect(_retrieved(chunk_a, chunk_b))

    assert result.has_contradiction is False


def test_detect_degrades_gracefully_when_the_judge_llm_call_fails():
    chunk_a = make_chunk("doc_a", text="İade süresi 14 gündür.")
    chunk_b = make_chunk("doc_b", text="İade süresi 30 gündür.")
    embedder = _ScriptedEmbeddings({chunk_a.text: [1.0, 0.0], chunk_b.text: [1.0, 0.0]})
    judge = _ScriptedLLM(raises=RuntimeError("judge unreachable"))

    detector = ContradictionDetector(embedder, judge)
    result = detector.detect(_retrieved(chunk_a, chunk_b))

    assert result.has_contradiction is False
    assert result.warning_message is None


def test_detect_degrades_gracefully_on_unparsable_judge_output():
    chunk_a = make_chunk("doc_a", text="İade süresi 14 gündür.")
    chunk_b = make_chunk("doc_b", text="İade süresi 30 gündür.")
    embedder = _ScriptedEmbeddings({chunk_a.text: [1.0, 0.0], chunk_b.text: [1.0, 0.0]})
    judge = _ScriptedLLM("not json at all")

    detector = ContradictionDetector(embedder, judge)
    result = detector.detect(_retrieved(chunk_a, chunk_b))

    assert result.has_contradiction is False


def test_detect_empty_chunk_list_returns_false_without_calling_embedder():
    calls: list[list[str]] = []

    class _CountingEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            return []

        def embed_query(self, text: str) -> list[float]:
            raise AssertionError("not used")

    detector = ContradictionDetector(_CountingEmbeddings(), _ScriptedLLM())
    result = detector.detect([])

    assert result.has_contradiction is False
    assert calls == []


def test_detect_single_chunk_returns_false_without_calling_embedder():
    calls: list[list[str]] = []

    class _CountingEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            return []

        def embed_query(self, text: str) -> list[float]:
            raise AssertionError("not used")

    chunk = make_chunk("doc_a", text="İade süresi 14 gündür.")
    detector = ContradictionDetector(_CountingEmbeddings(), _ScriptedLLM())
    result = detector.detect(_retrieved(chunk))

    assert result.has_contradiction is False
    assert calls == []
