from __future__ import annotations

import math

import pytest

from kuhaku.evaluation._tokenization import tokenize
from kuhaku.evaluation.metrics import (
    answer_correctness,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_partial_match():
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=3) == 0.5


def test_recall_at_k_full_match():
    assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_recall_at_k_empty_retrieved_is_zero():
    assert recall_at_k([], {"a"}, k=5) == 0.0


def test_recall_at_k_no_overlap_is_zero():
    assert recall_at_k(["x", "y"], {"a"}, k=5) == 0.0


def test_recall_at_k_uses_list_length_when_k_too_large():
    assert recall_at_k(["a"], {"a"}, k=99) == 1.0


def test_precision_at_k_basic():
    assert precision_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1 / 3)


def test_precision_at_k_k_larger_than_list_uses_list_length():
    assert precision_at_k(["a", "x"], {"a"}, k=10) == 0.5


def test_precision_at_k_empty_retrieved_is_zero():
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_mrr_first_hit_at_rank_two():
    assert mrr(["x", "a", "b"], {"a"}) == 0.5


def test_mrr_no_hit_is_zero():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_mrr_empty_retrieved_is_zero():
    assert mrr([], {"a"}) == 0.0


def test_hit_rate_at_k_hit():
    assert hit_rate_at_k(["x", "a"], {"a"}, k=2) == 1.0


def test_hit_rate_at_k_miss():
    assert hit_rate_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_hit_rate_at_k_empty_retrieved_is_zero():
    assert hit_rate_at_k([], {"a"}, k=5) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_at_k_reversed_ranking_below_one():
    score = ndcg_at_k(["b", "a"], {"a"}, k=2)
    expected_dcg = 1.0 / math.log2(3)
    expected_idcg = 1.0 / math.log2(2)
    assert score == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_at_k_empty_retrieved_is_zero():
    assert ndcg_at_k([], {"a"}, k=5) == 0.0


def test_ndcg_at_k_no_overlap_is_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_answer_correctness_identical_is_one():
    assert answer_correctness("the quick fox", "the quick fox") == 1.0


def test_answer_correctness_no_overlap_is_zero():
    assert answer_correctness("abc def", "xyz uvw") == 0.0


def test_answer_correctness_partial_overlap():
    score = answer_correctness("a b c", "a b d")
    assert score == pytest.approx(2 / 4)


def test_answer_correctness_empty_strings_are_zero():
    assert answer_correctness("", "golden") == 0.0
    assert answer_correctness("generated", "") == 0.0


def test_tokenize_behavior():
    assert tokenize("Hello World") == {"hello", "world"}
    assert tokenize("   ") == set()
    assert tokenize("") == set()
    assert tokenize("Foo  BAR\n\tbaz") == {"foo", "bar", "baz"}

