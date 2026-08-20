"""Tests for RAG-specific OTel metrics (kuhaku.tools.rag.metrics).

Split out of tests/observability/test_metrics.py (D57 cleanup): these instruments are
meaningful only to the RAG tool -- see tools/rag/metrics.py's own module docstring for
why they don't live in core/observability/metrics.py. Generic, tool-agnostic metrics
(retry, guard v2, auth, evaluation harness, ...) stay tested there.
"""

from __future__ import annotations

import prometheus_client

from tests.conftest import (
    prometheus_counter_value as _counter_value,
)
from tests.conftest import (
    prometheus_gauge_value as _gauge_value,
)
from kuhaku.tools.rag.metrics import (
    ABSTENTION_COUNT,
    ACTIVE_DOCUMENTS,
    CACHE_HITS,
    CACHE_MISSES,
    FAITHFULNESS_SCORE,
    HALLUCINATION_RATE,
    QUERY_REWRITE_TOTAL,
    RAG_CANARY_DETECTED,
    RAG_CONTRADICTION_DETECTED_TOTAL,
    RAG_EMBEDDING_MODEL_VERSION,
    RAG_LLM_MODEL_VERSION,
    RAG_PII_EGRESS,
    RAG_REPLAY_TOTAL,
    RAG_SYSTEM_PROMPT_VERSION,
    RAG_UNGROUNDED_CITATIONS,
    REQUEST_COUNT,
    RETRIEVER_STRATEGY,
    UNVERIFIED_CITATIONS,
    record_contradiction_detected,
    record_faithfulness,
    record_model_versions,
    record_query_rewrite,
    record_replay,
    record_unverified_citation,
)


# --- counters/gauges are wired correctly --------------------------------------
def test_request_count_increments_by_status():
    before = _counter_value(REQUEST_COUNT, status="ok")
    REQUEST_COUNT.add(1, {"status": "ok"})
    assert _counter_value(REQUEST_COUNT, status="ok") == before + 1


def test_retriever_strategy_counter():
    before = _counter_value(RETRIEVER_STRATEGY, strategy="hybrid")
    RETRIEVER_STRATEGY.add(1, {"strategy": "hybrid"})
    assert _counter_value(RETRIEVER_STRATEGY, strategy="hybrid") == before + 1


def test_abstention_count_has_reason_label_and_increments():
    before_zero = _counter_value(ABSTENTION_COUNT, reason="zero_chunks")
    before_low = _counter_value(ABSTENTION_COUNT, reason="low_confidence")
    ABSTENTION_COUNT.add(1, {"reason": "zero_chunks"})
    ABSTENTION_COUNT.add(1, {"reason": "low_confidence"})
    assert _counter_value(ABSTENTION_COUNT, reason="zero_chunks") == before_zero + 1
    assert _counter_value(ABSTENTION_COUNT, reason="low_confidence") == before_low + 1


def test_active_documents_gauge_has_no_labels_and_is_settable():
    ACTIVE_DOCUMENTS.set(7)
    assert _gauge_value(ACTIVE_DOCUMENTS) == 7


# --- Category 2: cache, citation verification ------------------------------------
def test_cache_hits_counter_increments():
    before = _counter_value(CACHE_HITS)
    CACHE_HITS.add(1)
    assert _counter_value(CACHE_HITS) == before + 1


def test_cache_misses_counter_increments():
    before = _counter_value(CACHE_MISSES)
    CACHE_MISSES.add(1)
    assert _counter_value(CACHE_MISSES) == before + 1


def test_record_model_versions_sets_all_three_info_gauges():
    """D42: each gauge is labeled by the version string itself and set to 1 -- the
    Prometheus "info" convention for rarely-changing build/deployment metadata."""

    record_model_versions("llm-vX", "embed-vX", "prompt-vX")
    assert _gauge_value(RAG_LLM_MODEL_VERSION, version="llm-vX") == 1
    assert _gauge_value(RAG_EMBEDDING_MODEL_VERSION, version="embed-vX") == 1
    assert _gauge_value(RAG_SYSTEM_PROMPT_VERSION, version="prompt-vX") == 1


def test_record_unverified_citation_uses_the_raw_tag_as_the_label():
    before = _counter_value(UNVERIFIED_CITATIONS, citation="[S9]")
    record_unverified_citation(9)
    assert _counter_value(UNVERIFIED_CITATIONS, citation="[S9]") == before + 1


def test_record_unverified_citation_clamps_large_index_to_a_fixed_overflow_label():
    """Every value reaching this function is by definition out of range, with no natural
    ceiling (an LLM could echo an injected [S99999999]) -- see DECISIONS.md D38."""

    before = _counter_value(UNVERIFIED_CITATIONS, citation="[S:overflow]")
    record_unverified_citation(99_999_999)
    assert _counter_value(UNVERIFIED_CITATIONS, citation="[S:overflow]") == before + 1


# --- Prompt Injection Guard v2 output-side metrics (D39) ------------------------
def test_canary_detected_counter_increments():
    before = _counter_value(RAG_CANARY_DETECTED)
    RAG_CANARY_DETECTED.add(1)
    assert _counter_value(RAG_CANARY_DETECTED) == before + 1


def test_pii_egress_counter_increments():
    before = _counter_value(RAG_PII_EGRESS)
    RAG_PII_EGRESS.add(1)
    assert _counter_value(RAG_PII_EGRESS) == before + 1


def test_ungrounded_citations_counter_increments():
    before = _counter_value(RAG_UNGROUNDED_CITATIONS)
    RAG_UNGROUNDED_CITATIONS.add(1)
    assert _counter_value(RAG_UNGROUNDED_CITATIONS) == before + 1


# --- Evaluation: faithfulness/hallucination (D47) --------------------------------
def test_record_faithfulness_sets_both_gauges():
    record_faithfulness(0.9, 0.1)
    assert _gauge_value(FAITHFULNESS_SCORE) == 0.9
    assert _gauge_value(HALLUCINATION_RATE) == 0.1


# --- Query rewriting (D48) --------------------------------------------------------
def test_record_query_rewrite_increments_by_cache_hit_and_success():
    before = _counter_value(QUERY_REWRITE_TOTAL, cache_hit="false", success="true")
    record_query_rewrite(cache_hit="false", success="true")
    assert _counter_value(QUERY_REWRITE_TOTAL, cache_hit="false", success="true") == before + 1


# --- Hallucination replay (D49) ---------------------------------------------------
def test_record_replay_uses_match_as_the_label():
    before_match = _counter_value(RAG_REPLAY_TOTAL, match="true")
    before_mismatch = _counter_value(RAG_REPLAY_TOTAL, match="false")
    record_replay(True)
    record_replay(False)
    assert _counter_value(RAG_REPLAY_TOTAL, match="true") == before_match + 1
    assert _counter_value(RAG_REPLAY_TOTAL, match="false") == before_mismatch + 1


# --- Real-time contradiction detection (D50) --------------------------------------
def test_record_contradiction_detected_increments():
    before = _counter_value(RAG_CONTRADICTION_DETECTED_TOTAL)
    record_contradiction_detected()
    assert _counter_value(RAG_CONTRADICTION_DETECTED_TOTAL) == before + 1


# --- Prometheus exposition (D57) --------------------------------------------------
def _exposition_text() -> str:
    return prometheus_client.generate_latest().decode("utf-8")


def test_request_count_appears_in_prometheus_exposition_with_expected_name_and_label():
    REQUEST_COUNT.add(1, {"status": "ok"})
    text = _exposition_text()
    assert 'rag_requests_total{' in text
    assert 'status="ok"' in text


def test_no_label_counter_appears_with_a_single_series():
    CACHE_HITS.add(1)
    text = _exposition_text()
    assert "rag_cache_hits_total " in text or "rag_cache_hits_total{" in text
