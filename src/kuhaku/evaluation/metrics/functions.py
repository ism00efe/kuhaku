"""Reusable retrieval and answer-correctness scoring functions.

Definitions are kept consistent with the ad hoc scoring in
``scripts/benchmark_quick.py``/``benchmark_sprint.py`` and ``eval/evaluate.py``
(hit-in-top-k, first-relevant-rank reciprocal). Plain functions, independently testable
and reused by the ``BaseMetric`` classes in this package so the scoring math lives in
exactly one place.
"""

from __future__ import annotations

import math

from .._tokenization import tokenize


def _effective_k(retrieved_ids: list[str], k: int) -> int:
    """Clamp ``k`` to ``len(retrieved_ids)`` -- a caller asking for more than what was
    retrieved must not be penalized for it."""

    return min(k, len(retrieved_ids))


def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Fraction of ``expected_ids`` present in the top-``k`` of ``retrieved_ids``."""

    if not retrieved_ids or not expected_ids:
        return 0.0
    top_k = retrieved_ids[: _effective_k(retrieved_ids, k)]
    hits = len(expected_ids & set(top_k))
    return hits / len(expected_ids)


def precision_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Fraction of the top-``k`` retrieved ids that are relevant."""

    if not retrieved_ids or not expected_ids:
        return 0.0
    effective_k = _effective_k(retrieved_ids, k)
    top_k = retrieved_ids[:effective_k]
    hits = sum(1 for doc_id in top_k if doc_id in expected_ids)
    return hits / effective_k


def mrr(retrieved_ids: list[str], expected_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant id in ``retrieved_ids`` (0.0 if none)."""

    if not retrieved_ids or not expected_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k``, binary relevance."""

    if not retrieved_ids or not expected_ids:
        return 0.0
    effective_k = _effective_k(retrieved_ids, k)
    top_k = retrieved_ids[:effective_k]

    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(top_k, start=1) if doc_id in expected_ids
    )
    ideal_hits = min(len(expected_ids), effective_k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """1.0 if any of the top-``k`` retrieved ids is relevant, else 0.0."""

    if not retrieved_ids or not expected_ids:
        return 0.0
    top_k = retrieved_ids[: _effective_k(retrieved_ids, k)]
    return 1.0 if any(doc_id in expected_ids for doc_id in top_k) else 0.0


def answer_correctness(generated_answer: str, golden_answer: str) -> float:
    """Token-overlap (Jaccard) similarity between a generated and golden answer.

    A baseline fallback only -- an LLM-as-judge evaluator (see ``judges.py``) can
    replace this with a semantic score without changing the metric's interface.
    """

    generated_tokens = tokenize(generated_answer)
    golden_tokens = tokenize(golden_answer)
    if not generated_tokens or not golden_tokens:
        return 0.0
    intersection = generated_tokens & golden_tokens
    union = generated_tokens | golden_tokens
    return len(intersection) / len(union)
