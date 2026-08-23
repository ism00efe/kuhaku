"""RAG-specific OpenTelemetry metrics.

Split out of ``kuhaku.core.observability.metrics``: every instrument
here is meaningful only to the RAG tool (retrieval strategies, abstentions, the
query-answer cache, citation verification, output-guard detections, faithfulness/
hallucination, query rewriting, hallucination replay, contradiction detection, and the
deployed RAG model/prompt versions) -- generic core observability must stay tool-agnostic
(see ``core/observability/metrics.py``'s own module docstring). Reuses the same OTel
meter/gauge helpers ``metrics.py`` itself uses, straight from
``core.observability.telemetry`` -- never through ``core.observability.metrics``, since a
tool may depend on core but core must never depend on a tool.
"""

from __future__ import annotations

from kuhaku.core.observability.telemetry import create_settable_gauge, get_meter

_meter = get_meter()

REQUEST_COUNT = _meter.create_counter(
    "rag_requests_total", unit="1", description="Total answer requests, by outcome"
)
RETRIEVER_STRATEGY = _meter.create_counter(
    "rag_retriever_strategy_total", unit="1", description="Requests handled, by retriever strategy"
)
ABSTENTION_COUNT = _meter.create_counter(
    "rag_abstention_total",
    unit="1",
    description=(
        "Answers withheld because retrieval returned zero chunks or low confidence "
        "(no LLM call made)"
    ),
)
ACTIVE_DOCUMENTS = create_settable_gauge(
    "rag_active_documents_count",
    description="Chunks passing the freshness filter in the most recent dense retrieval",
)

# --- Category 2: query-answer cache, citation verification ----------------------
CACHE_HITS = _meter.create_counter(
    "rag_cache_hits_total", unit="1", description="Query-answer cache hits (LLM call skipped)"
)
CACHE_MISSES = _meter.create_counter(
    "rag_cache_misses_total", unit="1", description="Query-answer cache misses (LLM call made)"
)
UNVERIFIED_CITATIONS = _meter.create_counter(
    "rag_unverified_citations_total",
    unit="1",
    description="LLM citations that did not match any retrieved source, by tag",
)

# Every index reaching this function is by definition invalid (a valid [S#] tag
# never gets here), and there is no natural ceiling at all: an LLM echoing injected
# document content could emit
# [S99999999]. Anything outside this range collapses to a fixed overflow label so the
# metric stays low-cardinality regardless of what the model outputs.
_MAX_LABELED_CITATION_TAG = 100


def record_unverified_citation(index: int) -> None:
    """Increment the unverified-citations counter for one invalid ``[S#]`` tag."""

    label = f"[S{index}]" if 0 <= index <= _MAX_LABELED_CITATION_TAG else "[S:overflow]"
    UNVERIFIED_CITATIONS.add(1, {"citation": label})


# --- Prompt Injection Guard v2 output-side metrics --------------------------------
# Only tools/rag/engine.py emits these three (canary/PII egress/ungrounded citations) --
# the guard v2 escalation/zone/degradation metrics they sit alongside in the guard
# pipeline itself are generic, tool-agnostic input-validation infrastructure and stay in
# core/observability/metrics.py (core.security.guard/classifier).
RAG_CANARY_DETECTED = _meter.create_counter(
    "rag_canary_detected_total",
    unit="1",
    description="LLM output contained the canary token (prompt extraction detected)",
)
RAG_PII_EGRESS = _meter.create_counter(
    "rag_pii_egress_total",
    unit="1",
    description="LLM output contained PII not present in any retrieved chunk (response blocked)",
)
RAG_UNGROUNDED_CITATIONS = _meter.create_counter(
    "rag_ungrounded_citations_total",
    unit="1",
    description="Citations flagged for low lexical overlap with their referenced source",
)


# --- Model and prompt versioning ---------------------------------------------------
# "Info" gauges (Prometheus convention for rarely-changing build/deployment metadata):
# each carries a single, low-cardinality `version` label and is always set to 1. Not set
# at module import time -- record_model_versions() is called once, from
# the embedding application's composition root, after settings are resolved there.
RAG_LLM_MODEL_VERSION = create_settable_gauge(
    "rag_llm_model_version", description="Deployed LLM version currently in use"
)
RAG_EMBEDDING_MODEL_VERSION = create_settable_gauge(
    "rag_embedding_model_version", description="Deployed embedding model version currently in use"
)
RAG_SYSTEM_PROMPT_VERSION = create_settable_gauge(
    "rag_system_prompt_version",
    description="Deployed (production) system prompt version currently in use",
)


def record_model_versions(
    llm_version: str, embedding_version: str, system_prompt_version: str
) -> None:
    """Set the three model/prompt version info-gauges.

    Idempotent: re-setting the same label to 1 is a no-op; a changed version simply
    activates a new label series, leaving the previous one's last value in place until
    the process restarts (acceptable here since a version change is always a redeploy,
    not a live value operators watch trend on).
    """

    RAG_LLM_MODEL_VERSION.set(1, {"version": llm_version})
    RAG_EMBEDDING_MODEL_VERSION.set(1, {"version": embedding_version})
    RAG_SYSTEM_PROMPT_VERSION.set(1, {"version": system_prompt_version})


# --- Evaluation metrics infrastructure ----------------------------------------------
# Plain (not "info") gauges: they track the most recently observed value from the async
# Faithfulness background task, not a rarely-changing deployment label -- last-observed
# value, not a true rolling average.
FAITHFULNESS_SCORE = create_settable_gauge(
    "rag_faithfulness_score",
    description="Most recently observed LLM-as-judge faithfulness score (0.0-1.0)",
)
HALLUCINATION_RATE = create_settable_gauge(
    "rag_hallucination_rate",
    description="Most recently observed hallucination rate (1 - faithfulness, 0.0-1.0)",
)


def record_faithfulness(score: float, hallucination_rate: float) -> None:
    """Set the faithfulness/hallucination-rate gauges."""

    FAITHFULNESS_SCORE.set(score)
    HALLUCINATION_RATE.set(hallucination_rate)


# --- Query rewriting -----------------------------------------------------------------
QUERY_REWRITE_TOTAL = _meter.create_counter(
    "rag_query_rewrite_total",
    unit="1",
    description="Pre-retrieval query rewrites performed, by cache hit and success",
)


def record_query_rewrite(cache_hit: str, success: str) -> None:
    """Increment the query-rewrite counter. Labels are ``"true"``/``"false"``."""

    QUERY_REWRITE_TOTAL.add(1, {"cache_hit": cache_hit, "success": success})


# --- Hallucination replay -------------------------------------------------------------
RAG_REPLAY_TOTAL = _meter.create_counter(
    "rag_replay_total",
    unit="1",
    description="Response replays performed via the embedding application's admin replay endpoint, by text match",
)


def record_replay(match: bool) -> None:
    """Increment the replay counter."""

    RAG_REPLAY_TOTAL.add(1, {"match": "true" if match else "false"})


# --- Real-time contradiction detection ------------------------------------------------
RAG_CONTRADICTION_DETECTED_TOTAL = _meter.create_counter(
    "rag_contradiction_detected_total",
    unit="1",
    description="Confirmed contradictions between retrieved chunks, detected at query time",
)


def record_contradiction_detected() -> None:
    """Increment the contradiction-detected counter, once per confirmed pair."""

    RAG_CONTRADICTION_DETECTED_TOTAL.add(1)
