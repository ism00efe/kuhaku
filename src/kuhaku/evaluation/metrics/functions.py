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


def _dedupe(retrieved_ids: list[str]) -> list[str]:
    """Keep each id at its first (best) rank only.

    Targets usually report one id per retrieved *chunk*, so a document with three chunks
    in the top-k would otherwise count three times -- inflating precision and pushing
    nDCG above 1.0, since the ideal ranking can only contain each relevant id once.
    """

    return list(dict.fromkeys(retrieved_ids))


def _top_k(retrieved_ids: list[str], k: int) -> list[str]:
    return _dedupe(retrieved_ids)[: max(k, 0)]


def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Fraction of ``expected_ids`` present in the top-``k`` distinct retrieved ids."""

    if not retrieved_ids or not expected_ids:
        return 0.0
    hits = len(expected_ids & set(_top_k(retrieved_ids, k)))
    return hits / len(expected_ids)


def precision_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Relevant ids among the top-``k`` distinct retrieved ids, divided by ``k``.

    Divided by the *requested* ``k``, not by how many ids came back: a target that
    returns one lucky id for k=5 retrieved 1/5 of what was asked for, and must not score
    the same as one that filled all five slots with relevant ids.
    """

    if not retrieved_ids or not expected_ids or k <= 0:
        return 0.0
    hits = sum(1 for doc_id in _top_k(retrieved_ids, k) if doc_id in expected_ids)
    return hits / k


def mrr(retrieved_ids: list[str], expected_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant id among the distinct retrieved ids."""

    if not retrieved_ids or not expected_ids:
        return 0.0
    for rank, doc_id in enumerate(_dedupe(retrieved_ids), start=1):
        if doc_id in expected_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k``, binary relevance.

    The ideal ranking holds ``min(len(expected_ids), k)`` relevant ids -- not clamped to
    how many ids the target returned, so a short list cannot reach 1.0 by returning less.
    Duplicate ids count once, at their first rank, so the score never exceeds 1.0.
    """

    if not retrieved_ids or not expected_ids or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(_top_k(retrieved_ids, k), start=1)
        if doc_id in expected_ids
    )
    ideal_hits = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def hit_rate_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """1.0 if any of the top-``k`` distinct retrieved ids is relevant, else 0.0."""

    if not retrieved_ids or not expected_ids:
        return 0.0
    return 1.0 if any(doc_id in expected_ids for doc_id in _top_k(retrieved_ids, k)) else 0.0


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
