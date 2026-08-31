"""Shared pytest fixtures and lightweight fakes.

The pipeline is programmed against small interfaces (EmbeddingProvider, VectorStore,
LLMProvider, Retriever, Reranker), so these in-memory fakes let us test the RAG engine,
retrieval chain, ingestion, and the application service deterministically — without
loading models, Chroma, or an LLM.
"""

from __future__ import annotations

import prometheus_client
import pytest

from kuhaku.core.auth import AuthContext
from kuhaku.core.security.classifier import Stage1Result, Stage2Result
from kuhaku.core.security.guard import GuardDecision
from kuhaku.tools.rag.models import ACCESS_TAGS_NONE_KEY, Chunk, RetrievedChunk
from kuhaku.tools.rag.retriever import is_entitled


@pytest.fixture(autouse=True)
def _isolate_kuhaku_decisions(tmp_path_factory, monkeypatch):
    """Point kuhaku's ``.kuhaku/decisions.json`` at a throwaway directory so a test that
    builds a real ``RAG()`` or calls ``build_llm_provider`` through the resolver never
    writes into the working tree (AGENTS.md: leave nothing behind).

    ``KUHAKU_PROJECT_DIR`` is read at ``JsonMemory()`` construction time, not import
    time, so an autouse fixture (which runs before every test body) is enough -- no
    import-order dependency."""

    monkeypatch.setenv("KUHAKU_PROJECT_DIR", str(tmp_path_factory.mktemp("kuhaku_decisions")))


def _matching_samples(instrument, family_name: str, sample_name: str, labels: dict[str, str]):
    # Prometheus label names may not contain a ".": the OTel Prometheus exporter
    # sanitizes attribute keys by replacing "." with "_" on export (verified against the
    # installed exporter -- e.g. the GenAI convention key "gen_ai.system" becomes the
    # Prometheus label "gen_ai_system"). Callers pass the real attribute key constants
    # (dots and all); this translates them the same way before matching.
    sanitized_labels = {k.replace(".", "_"): v for k, v in labels.items()}
    for family in prometheus_client.REGISTRY.collect():
        if family.name != family_name:
            continue
        for sample in family.samples:
            if sample.name != sample_name:
                continue
            if all(sample.labels.get(k) == v for k, v in sanitized_labels.items()):
                yield sample.value


def prometheus_counter_value(instrument, **labels: str) -> float:
    """Read a Counter's current exported value through the real Prometheus bridge --
    ``prometheus_client.REGISTRY``, the exact registry ``telemetry.py``'s
    ``PrometheusMetricReader`` populates and the real ``GET /api/admin/metrics`` route /
    ``metrics_summary.py`` both read. Deliberately not a synthetic OTel-only read: this
    doubles as a regression check that the OTel -> Prometheus bridge itself still works.

    Takes the instrument object itself (e.g. ``RETRY_ATTEMPTS``) so call sites read the
    same as they did against the old prometheus_client Counters. Sums every series
    matching the given (possibly partial) label set -- a counter can carry attribute
    dimensions beyond the ones a caller filters by (e.g. matching only ``gen_ai.system``
    without pinning ``gen_ai.request.model``), in which case more than one underlying
    series matches and a real total must add them, the same as a Prometheus query that
    doesn't group by every label would. Returns ``0.0`` when nothing matches -- same
    "unseen label = zero" behavior the old ``counter.labels(**labels)._value.get()`` had
    (prometheus_client auto-vivifies a zero child on first ``.labels()`` access).
    """

    family_name = instrument.name.removesuffix("_total")
    sample_name = f"{family_name}_total"
    return sum(_matching_samples(instrument, family_name, sample_name, labels))


def prometheus_histogram_count(instrument, **labels: str) -> float:
    """Sum a Histogram's ``_count`` samples matching ``labels`` (a subset match -- extra
    labels OTel adds, like ``otel_scope_name``, are ignored, same as
    :func:`prometheus_counter_value`)."""

    sample_name = f"{instrument.name}_count"
    return sum(_matching_samples(instrument, instrument.name, sample_name, labels))


def prometheus_gauge_value(instrument, **labels: str) -> float:
    """Read a Gauge's current exported value (no suffix stripping/appending -- gauges
    export under their own name unchanged, unlike Counters)."""

    return next(_matching_samples(instrument, instrument.name, instrument.name, labels), 0.0)


class FakeEmbeddings:
    """Deterministic stand-in for an embedding provider.

    Returns a tiny fixed-dimension vector; records calls so tests can assert prefixes
    or query text were passed through.
    """

    def __init__(self) -> None:
        self.doc_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [float(len(text)), 1.0, 0.0]


def _matches_where(chunk: Chunk, where: dict | None) -> bool:
    """Tiny interpreter for the small subset of Chroma ``where`` syntax
    :class:`FakeVectorStore` needs to support -- just enough to exercise
    ``DenseRetriever``'s entitlement push-down (see ``retriever._entitlement_where``)
    without a real Chroma instance. Operates directly on chunk attributes rather than a
    serialized metadata dict, since this fake never round-trips through storage.
    """

    if not where:
        return True
    if "$or" in where:
        return any(_matches_where(chunk, clause) for clause in where["$or"])
    ((key, cond),) = where.items()
    if key == ACCESS_TAGS_NONE_KEY:
        return not chunk.access_tags
    if key == "access_tags" and isinstance(cond, dict) and "$contains" in cond:
        return cond["$contains"] in chunk.access_tags
    raise ValueError(f"FakeVectorStore: unsupported where clause {where!r}")


class FakeVectorStore:
    """In-memory vector store honoring the VectorStore protocol.

    Ignores actual vector math: ``query`` returns the first ``top_k`` chunks matching
    ``where`` (see ``_matches_where``), which is enough to exercise retrieval, prompting,
    entitlement push-down, and citation wiring.
    """

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = list(chunks or [])
        self.reset_called = 0
        self.query_where_calls: list[dict | None] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._chunks.extend(chunks)

    def query(
        self, embedding: list[float], top_k: int, *, where: dict | None = None
    ) -> list[RetrievedChunk]:
        self.query_where_calls.append(where)
        candidates = [c for c in self._chunks if _matches_where(c, where)]
        return [
            RetrievedChunk(chunk=c, score=1.0 - 0.1 * i)
            for i, c in enumerate(candidates[:top_k])
        ]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self.reset_called += 1
        self._chunks = []

    def iter_chunks(self):
        return iter(self._chunks)


class FakeLLM:
    """LLM provider stub that returns a preset answer and records the last prompt."""

    def __init__(self, response: str = "Answer [S1].") -> None:
        self.response = response
        self.last_system: str | None = None
        self.last_user: str | None = None

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self.response


class FakeRetriever:
    """Returns a preset ranking, recording the query and depth it was asked for.

    Enforces document-level access filtering via the real ``is_entitled`` predicate
    (same rule ``DenseRetriever``/``BM25Retriever`` enforce), so engine-level tests that
    script this retriever get realistic entitlement behavior for tagged chunks -- for
    untagged chunks (every existing test's fixtures, before this feature existed) this
    is a no-op, since ``is_entitled`` always returns ``True`` for those.
    """

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks = list(chunks or [])
        self.calls: list[tuple[str, int]] = []
        self.auth_context_calls: list[AuthContext | None] = []
        self.doc_type_calls: list[str | None] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        auth_context: AuthContext | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        self.auth_context_calls.append(auth_context)
        self.doc_type_calls.append(doc_type)
        chunks = self._chunks
        if doc_type is not None:
            chunks = [c for c in chunks if c.doc_type == doc_type]
        chunks = [c for c in chunks if is_entitled(c, auth_context)]
        return [
            RetrievedChunk(chunk=c, score=1.0 - 0.1 * i)
            for i, c in enumerate(chunks[:top_k])
        ]


class FakeCache:
    """In-memory stand-in for QueryAnswerCache -- records calls, no TTL/SQLite."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.get_calls: list[str] = []
        self.put_calls: list[tuple[str, str]] = []

    def get(self, cache_key: str) -> str | None:
        self.get_calls.append(cache_key)
        return self._store.get(cache_key)

    def put(self, cache_key: str, answer_text: str) -> None:
        self.put_calls.append((cache_key, answer_text))
        self._store[cache_key] = answer_text


class FakeReranker:
    """Re-orders candidates by an explicit chunk-id preference list."""

    def __init__(self, preferred_order: list[str] | None = None) -> None:
        self.preferred_order = list(preferred_order or [])
        self.calls: list[tuple[str, int, int]] = []

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        self.calls.append((query, len(candidates), top_k))

        def rank_of(item: RetrievedChunk) -> int:
            cid = item.chunk.id
            return self.preferred_order.index(cid) if cid in self.preferred_order else 10**6

        ordered = sorted(candidates, key=rank_of)
        return ordered[:top_k]


_PASS_STAGE1 = Stage1Result(score=0.0, source="rule_based", top_features=[])
_NOT_RUN_STAGE2 = Stage2Result(ran=False, degraded=False, label=None, confidence=None)


def make_guard_decision(
    zone: str = "pass",
    *,
    stage1: Stage1Result | None = None,
    stage2: Stage2Result | None = None,
    escalation_reason: str | None = None,
    norm_drift: int = 0,
    normalized_length: int = 0,
) -> GuardDecision:
    return GuardDecision(
        zone=zone,
        stage1=stage1 or _PASS_STAGE1,
        stage2=stage2 or _NOT_RUN_STAGE2,
        escalation_reason=escalation_reason,
        norm_drift=norm_drift,
        normalized_length=normalized_length,
    )


class FakeGuardPipeline:
    """Scriptable stand-in for GuardPipeline -- returns a preset GuardDecision and
    records every query it was asked to evaluate."""

    def __init__(
        self,
        decision: GuardDecision | None = None,
        *,
        citation_grounding_threshold: float = 0.1,
        guard_version: str = "test-guard",
        model_version: str = "test-model",
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ) -> None:
        self.decision = decision or make_guard_decision()
        self.citation_grounding_threshold = citation_grounding_threshold
        self.guard_version = guard_version
        self.model_version = model_version
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.evaluate_calls: list[str] = []

    def evaluate(self, query: str) -> GuardDecision:
        self.evaluate_calls.append(query)
        return self.decision

    def validate(self) -> bool:
        """Startup self-check stand-in (kuhaku.core.policy.enforce_security_policy)."""

        self.evaluate("startup self-check")
        return True


def make_chunk(doc_id: str, index: int = 0, *, text: str = "content", title: str = "T",
               doc_type: str = "faq", content_type: str = "text",
               effective_date: str = "", obsolete: bool = False,
               expiry_date: str = "", access_tags: tuple[str, ...] = ()) -> Chunk:
    return Chunk(
        id=f"{doc_id}::{index}",
        document_id=doc_id,
        title=title,
        doc_type=doc_type,
        text=text,
        chunk_index=index,
        source_path=f"{doc_id}.md",
        content_type=content_type,
        effective_date=effective_date,
        obsolete=obsolete,
        expiry_date=expiry_date,
        access_tags=access_tags,
    )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
