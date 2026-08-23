"""The RAG engine: the heart of the pipeline.

Orchestrates the online query path:
    sanitize -> (optional) context summary -> guard -> guard v2 (opt-in) -> retrieve ->
    cache check -> prompt -> generate -> map citations -> flag unverified citations ->
    output guard v2 (opt-in) -> audit record (unconditional).

Depends only on the small interfaces (``EmbeddingProvider``, ``VectorStore``,
``LLMProvider``), never on concrete SDKs — so any of them can be swapped independently.

Every call is wrapped in a bound ``trace_id`` (see ``observability.bind_trace_id``) and
each stage is timed/logged/metriced via ``instrumented_step`` — nested layers (the
retriever's embed/rerank steps) pick the same trace_id up automatically through the
context var, with no change to their own signatures.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from kuhaku.core.auth import AuthContext
from kuhaku.core.llm.base import LLMError, LLMProvider, LLMUnavailableError
from kuhaku.core.observability import (
    bind_trace_id,
    instrumented_step,
    record_guard_stage1_escalation,
    record_guard_stage2_classification,
    record_guard_zone,
    record_redactions,
)
from kuhaku.core.sanitization import Redaction, sanitize_text
from kuhaku.core.security import (
    CANARY_TOKEN,
    GUARD_REJECT_MESSAGE,
    REFUSAL_MESSAGE,
    RESTRICTED_WARNING,
    GuardDecision,
    GuardPipeline,
    evaluate_output,
    inspect_query,
    record_audit,
)
from kuhaku.core.security.audit import AuditWriteError
from kuhaku.evaluation.sample import EvaluationSample
from kuhaku.tools.rag.models import Answer, Citation, RetrievedChunk

from .cache import QueryAnswerCache, compute_cache_key
from .chunking import Chunker, ParagraphChunker
from .config import RAGSettings
from .contradiction_detector import ContradictionDetector
from .embeddings import EmbeddingProvider
from .ingestion import ingest_single_document
from .context_summary import summarize_context
from .messages import DEFAULT_ENGINE_MESSAGES, EngineMessages
from .metrics import (
    ABSTENTION_COUNT,
    CACHE_HITS,
    CACHE_MISSES,
    RAG_CANARY_DETECTED,
    RAG_PII_EGRESS,
    RAG_UNGROUNDED_CITATIONS,
    REQUEST_COUNT,
    RETRIEVER_STRATEGY,
    record_contradiction_detected,
    record_unverified_citation,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .query_rewriter import QueryRewriter
from .retriever import DenseRetriever, Retriever, refresh_retriever
from .vectorstore import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"\[S(\d+)\]")

# Defense-in-depth cap on context_text length, independent of any length limit a caller
# may already apply upstream — protects any direct caller of `answer()`/`ask()`.
_MAX_CONTEXT_CHARS = 500_000


class RAGEngine:
    """Retrieval-augmented generation engine orchestrating retrieval, LLM synthesis, and guards."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
        *,
        top_k: int = 4,
        retriever: Retriever | None = None,
        input_guard_enabled: bool = True,
        chunker: Chunker | None = None,
        confidence_threshold: float = 0.15,
        cache: QueryAnswerCache | None = None,
        guard: GuardPipeline | None = None,
        audit_log_path: str = "",
        audit_enabled: bool = True,
        llm_version: str = "",
        embedding_version: str = "",
        system_prompt_version: str = "",
        system_prompt: str | None = None,
        query_rewriter: QueryRewriter | None = None,
        contradiction_detector: ContradictionDetector | None = None,
        contradiction_db_path: str | None = None,
        messages: EngineMessages | None = None,
        contradiction_storage: object | None = None,
        rag_settings: RAGSettings | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._top_k = top_k
        # Feature 1 (kuhaku tech-debt cleanup): optional RAG-scoped settings
        # projection, threaded down to ingest_document()'s doc-type inference. None
        # (the default) leaves every existing call site of RAGEngine(...) unaffected --
        # ingestion falls back to the process-global get_settings() the same way it
        # always has (see ingestion.py's _infer_doc_type).
        self._rag_settings = rag_settings
        # Default to plain dense retrieval; the composition root injects a
        # HybridRetriever when sparse/re-ranking are enabled.
        self._retriever = retriever or DenseRetriever(embedder, store)
        self._input_guard_enabled = input_guard_enabled
        self._chunker = chunker or ParagraphChunker()
        self._confidence_threshold = confidence_threshold
        # Optional query-answer cache. None (the default) disables caching
        # entirely -- every existing call site of RAGEngine(...) is unaffected.
        self._cache = cache
        # Prompt Injection Guard v2: optional, dormant unless configured (mirrors
        # `cache` above). None (the default) leaves every existing call site and every
        # existing behavior byte-identical -- the legacy `input_guard_enabled` path above
        # is completely independent of this.
        self._guard = guard
        self._audit_log_path = audit_log_path
        # Mirrors Settings.audit_enabled -- threaded through to every
        # record_audit() call site below so a disabled audit log is an immediate,
        # filesystem-untouched no-op regardless of which call site would have written.
        self._audit_enabled = audit_enabled
        # Deployed versions from Settings, threaded in by service.build_service().
        # Default "" (not required) so direct construction -- every test in this suite,
        # eval scripts -- keeps working unchanged, mirroring `auth_context`'s own
        # default-None precedent for the same reason.
        self._llm_version = llm_version
        self._embedding_version = embedding_version
        self._system_prompt_version = system_prompt_version
        # Instance-scoped prompt text, defaulting to the module-level constant so
        # every existing call site (which never passes this) is byte-identical to before.
        # AssistantService.reconfigure() mutates this attribute directly to hot-reload an
        # edited prompt without rebuilding the engine.
        self._system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        # Optional pre-retrieval query rewriter. None (the default) disables
        # rewriting entirely -- retrieval uses `retrieval_query` unchanged, exactly as
        # before this feature existed. When set, only the retriever call sees the
        # rewritten text; the guard, the QA-cache key, and the generation prompt all
        # keep using the original `retrieval_query`.
        self._query_rewriter = query_rewriter
        # Optional post-retrieval contradiction check. None (the default) disables
        # detection entirely -- every existing call site of RAGEngine(...) is unaffected.
        # `contradiction_db_path` is where confirmed contradictions get logged
        # (evaluation/contradiction_storage.py); it is independent of `audit_log_path`
        # since the JSONL audit log and the SQLite evaluation DB are separate stores.
        self._contradiction_detector = contradiction_detector
        self._contradiction_db_path = contradiction_db_path
        # Optional contradiction storage hook. When set (by the application layer),
        # this callable receives the same args as the old inline
        # ``contradiction_storage.log_contradiction(...)`` call and persists the pairs to
        # the evaluation DB. ``None`` (the default) disables persistence entirely so the
        # kuhaku has no compile-time or runtime dependency on the application layer.
        self._contradiction_storage = contradiction_storage
        # User-facing strings are injectable: the engine's control flow stays free of
        # any hard-coded copy. Callers that pass nothing get DEFAULT_ENGINE_MESSAGES.
        self._messages = messages if messages is not None else DEFAULT_ENGINE_MESSAGES

    def _require_methods(self, obj: object, methods: tuple[str, ...], param: str) -> None:
        """Duck-type guard for Protocol-typed parameters (Protocols aren't safely
        ``isinstance``-checkable at runtime unless declared ``@runtime_checkable``, which
        would still only check method *names*, not signatures -- so this checks the same
        thing explicitly and raises a message naming the offending parameter)."""

        missing = [m for m in methods if not callable(getattr(obj, m, None))]
        if missing:
            raise TypeError(
                f"{param} must implement {', '.join(methods)}; "
                f"got {type(obj).__name__} missing {', '.join(missing)}"
            )

    def update_llm(self, llm: LLMProvider) -> None:
        """Replace the LLM provider used for answer generation."""

        if not isinstance(llm, LLMProvider):
            raise TypeError(f"llm must implement LLMProvider; got {type(llm).__name__}")
        self._llm = llm

    def update_embedder(self, embedder: EmbeddingProvider) -> None:
        """Replace the embedding provider used for retrieval and ingestion."""

        if not isinstance(embedder, EmbeddingProvider):
            raise TypeError(
                f"embedder must implement EmbeddingProvider; got {type(embedder).__name__}"
            )
        self._embedder = embedder

    def update_vector_store(self, store: VectorStore) -> None:
        """Replace the vector store backing retrieval and ingestion.

        ``VectorStore`` is a plain (non-``runtime_checkable``) ``Protocol``, so this
        duck-types the methods ``retrieve``/``answer`` actually call instead of using
        ``isinstance`` (which ``Protocol`` only supports when decorated
        ``@runtime_checkable`` -- see ``LLMProvider``/``EmbeddingProvider`` above).
        """

        self._require_methods(store, ("add", "query", "count", "reset"), "store")
        self._store = store

    def update_retriever(self, retriever: Retriever) -> None:
        """Replace the retriever used by :meth:`retrieve`/:meth:`answer`."""

        if not isinstance(retriever, Retriever):
            raise TypeError(
                f"retriever must implement Retriever; got {type(retriever).__name__}"
            )
        self._retriever = retriever

    def update_chunker(self, chunker: Chunker) -> None:
        """Replace the chunker used by :meth:`ingest_document`."""

        self._require_methods(chunker, ("chunk",), "chunker")
        self._chunker = chunker

    def update_query_rewriter(self, rewriter: QueryRewriter | None) -> None:
        """Replace the pre-retrieval query rewriter, or ``None`` to disable it."""

        if rewriter is not None:
            self._require_methods(rewriter, ("rewrite",), "rewriter")
        self._query_rewriter = rewriter

    def update_contradiction_detector(self, detector: ContradictionDetector | None) -> None:
        """Replace the post-retrieval contradiction detector, or ``None`` to disable it."""

        if detector is not None:
            self._require_methods(detector, ("detect",), "detector")
        self._contradiction_detector = detector

    def update_system_prompt(self, prompt_text: str) -> None:
        """Hot-reload the system prompt text used for generation."""

        if not isinstance(prompt_text, str):
            raise TypeError(f"prompt_text must be a str; got {type(prompt_text).__name__}")
        self._system_prompt = prompt_text

    def update_system_prompt_version(self, version: str) -> None:
        """Update the system prompt version label recorded on every ``Answer``.

        Typed ``str`` (not ``int``) to match the existing ``"v3"``-style version labels
        this engine has always used (see ``__init__``'s ``system_prompt_version``
        parameter and ``service._bump_prompt_version``).
        """

        if not isinstance(version, str):
            raise TypeError(f"version must be a str; got {type(version).__name__}")
        self._system_prompt_version = version

    def update_top_k(self, k: int) -> None:
        """Replace the default number of chunks retrieved per query."""

        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError(f"k must be an int; got {type(k).__name__}")
        self._top_k = k

    def update_confidence_threshold(self, threshold: float) -> None:
        """Replace the minimum top-chunk score required to avoid abstaining."""

        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError(f"threshold must be a float; got {type(threshold).__name__}")
        self._confidence_threshold = float(threshold)

    def update_cache(self, cache: QueryAnswerCache | None) -> None:
        """Replace the query-answer cache, or ``None`` to disable caching."""

        if cache is not None:
            self._require_methods(cache, ("get", "put"), "cache")
        self._cache = cache

    def update_guard(self, guard: GuardPipeline | None) -> None:
        """Replace the Guard v2 pipeline, or ``None`` to disable it."""

        if guard is not None:
            self._require_methods(guard, ("evaluate",), "guard")
        self._guard = guard

    def get_system_prompt(self) -> str:
        """The current system prompt text (the hot-reload target)."""

        return self._system_prompt

    def get_system_prompt_version(self) -> str:
        """The current system prompt version label."""

        return self._system_prompt_version

    def get_llm(self) -> LLMProvider:
        """The LLM provider currently used for generation."""

        return self._llm

    def get_embedder(self) -> EmbeddingProvider:
        """The embedding provider currently used for retrieval and ingestion."""

        return self._embedder

    def get_vector_store(self) -> VectorStore:
        """The vector store currently backing retrieval and ingestion."""

        return self._store

    def get_retriever(self) -> Retriever:
        """The retriever currently used by :meth:`retrieve`/:meth:`answer`."""

        return self._retriever

    def get_chunker(self) -> Chunker:
        """The chunker currently used by :meth:`ingest_document`."""

        return self._chunker

    def get_query_rewriter(self) -> QueryRewriter | None:
        """The active pre-retrieval query rewriter, or ``None`` if disabled."""

        return self._query_rewriter

    def get_contradiction_detector(self) -> ContradictionDetector | None:
        """The active post-retrieval contradiction detector, or ``None`` if disabled."""

        return self._contradiction_detector

    def get_cache(self) -> QueryAnswerCache | None:
        """The active query-answer cache, or ``None`` if disabled."""

        return self._cache

    def get_guard(self) -> GuardPipeline | None:
        """The active Guard v2 pipeline, or ``None`` if disabled."""

        return self._guard

    def get_top_k(self) -> int:
        """The default number of chunks retrieved per query."""

        return self._top_k

    def get_confidence_threshold(self) -> float:
        """The minimum top-chunk score required to avoid abstaining."""

        return self._confidence_threshold

    def get_llm_version(self) -> str:
        """The deployed LLM version label recorded on every ``Answer``."""

        return self._llm_version

    def get_embedding_version(self) -> str:
        """The deployed embedding model version label recorded on every ``Answer``."""

        return self._embedding_version

    def get_input_guard_enabled(self) -> bool:
        """Whether the legacy (v1) input guard is active."""

        return self._input_guard_enabled

    def corpus_size(self) -> int:
        """Number of indexed chunks currently searchable."""

        return self._store.count()

    def ingest_document(
        self,
        text: str,
        filename: str,
        *,
        chunk_size: int,
        overlap: int,
        access_tags: Sequence[str] | None = None,
    ) -> tuple[int, list[Redaction]]:
        """Sanitize, chunk, embed, and add one operator-uploaded document.

        Delegates to :func:`ingest_single_document` using this engine's own embedder and
        store — the same instances ``retrieve``/``answer`` already search — so an upload
        is visible to the very next query without reloading any model or reconnecting to
        Chroma. Never resets the store; corpus documents loaded at startup are untouched.

        The configured upload size limit (``self._rag_settings.max_upload_bytes``, ``None``
        when no ``rag_settings`` was supplied) and this engine's ``EngineMessages`` are
        both threaded through, so a too-large upload is rejected with an injectable error
        instead of being silently indexed.

        ``access_tags``, when given, gates every chunk this document produces (document-
        level access filtering; see ``retriever.is_entitled``). ``None`` (the default)
        leaves the document untagged -- visible to any caller, unchanged from before this
        parameter existed.

        Also notifies ``self._retriever`` a new chunk may exist (see
        ``retriever.refresh_retriever``), so a sparse or hybrid retriever's cached BM25
        index picks up the new document on the next query -- without this engine ever
        needing to know it's talking to one.
        """

        max_upload_bytes = (
            self._rag_settings.max_upload_bytes if self._rag_settings is not None else None
        )
        result = ingest_single_document(
            text, filename, self._embedder, self._store,
            chunk_size=chunk_size, overlap=overlap, chunker=self._chunker,
            rag_settings=self._rag_settings, max_upload_bytes=max_upload_bytes,
            messages=self._messages, access_tags=access_tags,
        )
        chunks_added, _redactions = result
        if chunks_added:
            refresh_retriever(self._retriever)
        return result

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        auth_context: AuthContext | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for an already-sanitized query.

        ``auth_context`` is threaded straight to the retriever, which enforces
        document-level access filtering unconditionally (see
        ``kuhaku.tools.rag.retriever.is_entitled``) -- ``auth_context=None`` means "no
        roles", not "no filtering": only untagged chunks are visible. There is no way to
        disable this from here or anywhere else. ``doc_type`` (4.1) defaults to ``None``
        -- no filtering, identical to behavior before this parameter existed.
        """

        return self._retriever.retrieve(
            query, top_k or self._top_k, auth_context=auth_context, doc_type=doc_type
        )

    def answer(
        self,
        question: str,
        context_text: str | None = None,
        trace_id: str | None = None,
        *,
        auth_context: AuthContext | None = None,
    ) -> Answer:
        """Answer a question, optionally grounded by supplementary structured context.

        ``context_text`` is an optional blob (JSON, XML, or raw text) a caller supplies
        alongside the question -- e.g. something a user pasted or uploaded -- whose
        salient fields are extracted and folded into the retrieval query (see
        ``context_summary.summarize_context``).

        ``trace_id`` may be supplied by the caller (e.g. a future API layer); otherwise
        one is generated and bound for the duration of this call. ``auth_context`` (see
        ``kuhaku.core.auth``) comes from the authenticated caller when the API layer has
        one -- retrieval enforces document-level access filtering against it
        unconditionally (see ``kuhaku.tools.rag.retriever.is_entitled``); a caller
        passing ``None`` (the default, for unauthenticated internal use or direct engine
        construction in tests/eval scripts) is treated as having no roles, so only
        untagged chunks are visible to it -- there is no separate switch to disable this.
        ``auth_context`` also reaches every audit record written for this request, so the
        trail shows who accessed what without kuhaku assuming any fixed role or
        clearance vocabulary.
        """

        with bind_trace_id(trace_id) as tid:
            with instrumented_step("answer", trace_id=tid) as rec:
                result = self._answer(question, context_text, tid, auth_context=auth_context)
                rec.set(abstained=result.abstained)
                return result

    def evaluate_sample(self, query: str) -> EvaluationSample:
        """Implements ``EvaluationTarget`` (see ``evaluation/target.py``): runs the full
        ``answer()`` pipeline -- sanitization, guards, retrieval, generation, citation
        mapping -- and reduces the result to a tool-agnostic ``EvaluationSample`` an
        ``EvaluationRunner`` can score without knowing anything about ``RetrievedChunk``
        or ``Answer``."""

        result = self.answer(query)
        return EvaluationSample(
            query=query,
            answer=result.text,
            contexts=[item.chunk.text for item in result.retrieved],
            retrieved_doc_ids=[item.chunk.document_id for item in result.retrieved],
            metadata={"trace_id": result.trace_id, "abstained": result.abstained},
        )

    def _write_audit(
        self,
        *,
        trace_id: str,
        raw_question: str,
        sanitized_retrieval_query: str,
        event_type: str,
        auth_context: AuthContext | None,
        accessed_chunks: list[str],
        decision: GuardDecision | None = None,
        guard_version: str | None = None,
        model_version: str | None = None,
        thresholds: tuple[float, float] | None = None,
        output_checks: dict[str, object] | None = None,
    ) -> None:
        """One audit record for one request outcome (extended to every exit of
        ``_answer``, not just the two call sites originally covered).

        ``event_type`` names the outcome (mirrors the ``REQUEST_COUNT`` status label at
        the same exit, e.g. ``"empty_kb"``, ``"ok"``) so the audit trail can answer "what
        happened", not just "who asked". Always ids/counts (``accessed_chunks``), never
        chunk text or access tags -- same restriction the pre-existing successful-path
        call already observed.

        ``record_audit`` deliberately raises ``AuditWriteError`` while enabled so
        an operator notices a broken audit sink -- but the record is written *for* the
        operator, never read by the caller, so a broken sink must not turn an otherwise-
        successful answer into a failed request. The write failure is still visible: the
        exception is logged (with the metric ``record_audit`` itself already emitted
        before raising), just not propagated past this point.
        """

        try:
            record_audit(
                self._audit_log_path,
                enabled=self._audit_enabled,
                trace_id=trace_id,
                raw_question=raw_question,
                sanitized_retrieval_query=sanitized_retrieval_query,
                event_type=event_type,
                decision=decision,
                guard_version=guard_version,
                model_version=model_version,
                thresholds=thresholds,
                output_checks=output_checks,
                auth_context=auth_context,
                accessed_chunks=accessed_chunks,
                llm_version=self._llm_version,
                embedding_version=self._embedding_version,
                system_prompt_version=self._system_prompt_version,
            )
        except AuditWriteError:
            logger.error(
                "audit record could not be written; answer unaffected",
                extra={"event_type": event_type, "trace_id": trace_id},
                exc_info=True,
            )

    def _answer(
        self,
        question: str,
        context_text: str | None,
        trace_id: str,
        *,
        auth_context: AuthContext | None,
    ) -> Answer:
        if context_text and len(context_text) > _MAX_CONTEXT_CHARS:
            context_text = context_text[:_MAX_CONTEXT_CHARS]

        # 1) SECURITY: sanitize every user-supplied input before it goes anywhere.
        with instrumented_step("sanitize") as rec:
            clean_question, q_red = sanitize_text(question or "")
            redaction_labels = [f"{r.label}×{r.count}" for r in q_red]
            record_redactions(q_red)

            context_summary = ""
            if context_text:
                # SECURITY: sanitize the RAW context first (the guaranteed gate), then
                # summarize the already-clean text. This does not rely on the
                # summarizer's field allowlist to keep PII out, and it lets us report
                # what was actually masked.
                clean_context, c_red = sanitize_text(context_text)
                redaction_labels += [f"{r.label}×{r.count}" for r in c_red]
                record_redactions(c_red)
                context_summary = summarize_context(clean_context)
            rec.set(redaction_count=len(redaction_labels))

        # 2) Build the retrieval query (question + salient context fields).
        retrieval_query = clean_question.strip()
        if context_summary:
            label = self._messages.context_label
            retrieval_query = (
                f"{retrieval_query}\n{label} {context_summary}"
                if retrieval_query
                else f"{label} {context_summary}"
            )
        if not retrieval_query:
            REQUEST_COUNT.add(1, {"status": "empty"})
            self._write_audit(
                trace_id=trace_id,
                raw_question=question or "",
                sanitized_retrieval_query=retrieval_query,
                event_type="empty",
                auth_context=auth_context,
                accessed_chunks=[],
            )
            return Answer(
                text=self._messages.empty_query,
                citations=[],
                retrieved=[],
                redactions=redaction_labels,
                trace_id=trace_id,
            )

        # 3) SECURITY: block prompt-injection attempts before retrieval or the LLM ever
        # see the query. Runs on the already-sanitized text, so nothing logged here can
        # contain raw PII.
        if self._input_guard_enabled:
            with instrumented_step("input_guard") as rec:
                safe, reason = inspect_query(retrieval_query)
                rec.set(safe=safe)
            if not safe:
                logger.warning(
                    "query blocked by input guard",
                    extra={"reason": reason, "status": "blocked"},
                )
                REQUEST_COUNT.add(1, {"status": "blocked"})
                self._write_audit(
                    trace_id=trace_id,
                    raw_question=question or "",
                    sanitized_retrieval_query=retrieval_query,
                    event_type="blocked",
                    auth_context=auth_context,
                    accessed_chunks=[],
                )
                return Answer(
                    text=REFUSAL_MESSAGE,
                    citations=[],
                    retrieved=[],
                    redactions=redaction_labels,
                    trace_id=trace_id,
                )

        # 3b) SECURITY (v2, opt-in): normalize -> two-stage classify -> 3-zone decide.
        # Runs after the legacy guard above, which stays live and unconditional -- v2
        # only ever sees what already got past it; pure defense-in-depth, never a
        # replacement. Dormant unless `guard` was configured (Settings.guard_enabled).
        guard_decision: GuardDecision | None = None
        if self._guard is not None:
            with instrumented_step("guard_v2") as rec:
                guard_decision = self._guard.evaluate(retrieval_query)
                rec.set(
                    guard_zone=guard_decision.zone, stage1_score=guard_decision.stage1.score
                )
            record_guard_zone(guard_decision.zone)
            if guard_decision.escalation_reason:
                record_guard_stage1_escalation(guard_decision.escalation_reason)
            if guard_decision.stage2.ran and guard_decision.stage2.label is not None:
                record_guard_stage2_classification(guard_decision.stage2.label)

            if guard_decision.zone == "reject":
                logger.warning(
                    "query rejected by guard v2",
                    extra={"status": "guard_v2_reject", "guard_zone": "reject"},
                )
                REQUEST_COUNT.add(1, {"status": "guard_v2_reject"})
                self._write_audit(
                    trace_id=trace_id,
                    raw_question=question or "",
                    sanitized_retrieval_query=retrieval_query,
                    event_type="guard_v2_reject",
                    decision=guard_decision,
                    guard_version=self._guard.guard_version,
                    model_version=self._guard.model_version,
                    thresholds=(self._guard.low_threshold, self._guard.high_threshold),
                    auth_context=auth_context,
                    accessed_chunks=[],  # retrieval never ran -- rejected before it
                )
                return Answer(
                    text=GUARD_REJECT_MESSAGE,
                    citations=[],
                    retrieved=[],
                    redactions=redaction_labels,
                    trace_id=trace_id,
                    abstained=True,
                )

        # 4) Retrieve.
        try:
            corpus_is_empty = self._store.count() == 0
        except Exception as exc:
            raise VectorStoreError(f"Vector store count failed: {exc}") from exc
        if corpus_is_empty:
            REQUEST_COUNT.add(1, {"status": "empty_kb"})
            self._write_audit(
                trace_id=trace_id,
                raw_question=question or "",
                sanitized_retrieval_query=retrieval_query,
                event_type="empty_kb",
                auth_context=auth_context,
                accessed_chunks=[],
            )
            return Answer(
                text=self._messages.empty_kb,
                citations=[],
                retrieved=[],
                redactions=redaction_labels,
                trace_id=trace_id,
            )

        # Rewrite the query for the retriever ONLY -- `retrieval_query` itself
        # (already used above by the guard, and used below for the QA-cache key and
        # the generation prompt) is never replaced.
        search_query = retrieval_query
        if self._query_rewriter is not None:
            with instrumented_step("query_rewrite") as rec:
                search_query = self._query_rewriter.rewrite(retrieval_query)
                rec.set(rewritten=search_query != retrieval_query)

        with instrumented_step("retrieve") as rec:
            retrieved = self.retrieve(search_query, auth_context=auth_context)
            strategy = getattr(self._retriever, "strategy", "custom")
            rec.set(chunk_count=len(retrieved), strategy=strategy)
        RETRIEVER_STRATEGY.add(1, {"strategy": strategy})

        # Retrieval ran but found nothing relevant -- abstain rather than let the
        # LLM improvise an ungrounded answer from an empty prompt.
        if not retrieved:
            # SECURITY: an empty result is reported identically whether entitlement
            # filtering removed every candidate or the query genuinely matched nothing --
            # distinguishing the two would let a caller map restricted content by varying
            # the query and watching which empty results are "denials".
            logger.info("no chunks retrieved for query; abstaining", extra={"status": "no_chunks"})
            REQUEST_COUNT.add(1, {"status": "no_chunks"})
            ABSTENTION_COUNT.add(1, {"reason": "zero_chunks"})
            self._write_audit(
                trace_id=trace_id,
                raw_question=question or "",
                sanitized_retrieval_query=retrieval_query,
                event_type="no_chunks",
                auth_context=auth_context,
                accessed_chunks=[],
            )
            return Answer(
                text=self._messages.no_chunks,
                citations=[],
                retrieved=[],
                redactions=redaction_labels,
                trace_id=trace_id,
                abstained=True,
            )

        top_score = max(c.score for c in retrieved)
        if top_score < self._confidence_threshold:
            logger.info(
                "retrieved chunks below confidence threshold; abstaining",
                extra={
                    "status": "low_confidence",
                    "top_score": top_score,
                    "threshold": self._confidence_threshold,
                },
            )
            ABSTENTION_COUNT.add(1, {"reason": "low_confidence"})
            # No REQUEST_COUNT status label existed for this exit before this feature --
            # "low_confidence" is the one chosen here, used only as the audit event_type
            # (mirrors ABSTENTION_COUNT's own "low_confidence" reason above);
            # REQUEST_COUNT itself is left as-is, out of this feature's scope.
            self._write_audit(
                trace_id=trace_id,
                raw_question=question or "",
                sanitized_retrieval_query=retrieval_query,
                event_type="low_confidence",
                auth_context=auth_context,
                accessed_chunks=[c.chunk.id for c in retrieved],
            )
            return Answer(
                text=self._messages.no_chunks,
                citations=[],
                retrieved=[],
                redactions=redaction_labels,
                trace_id=trace_id,
                abstained=True,
            )

        retrieved = [c for c in retrieved if c.score >= self._confidence_threshold]

        # 4.5) Contradiction detection -- best-effort, never blocks the response.
        # Runs only on the current query's retrieved chunk set (Constraint 1), before
        # generation, so its [S#] references match the citation numbering the model's
        # response will use. Dormant unless a detector was configured
        # (RAGSettings.contradiction_detection_enabled).
        contradiction_warning: str | None = None
        if self._contradiction_detector is not None:
            try:
                contradiction_result = self._contradiction_detector.detect(retrieved)
                if contradiction_result.has_contradiction:
                    contradiction_warning = contradiction_result.warning_message
                    contradiction_pairs = contradiction_result.contradiction_pairs
                    logger.info(
                        "contradiction_detected",
                        extra={
                            "event": "contradiction_detected",
                            "trace_id": trace_id,
                            "pairs": len(contradiction_pairs),
                            "warning": contradiction_warning,
                        },
                    )
                    for _pair in contradiction_pairs:
                        record_contradiction_detected()
                    record_audit(
                        self._audit_log_path,
                        enabled=self._audit_enabled,
                        trace_id=trace_id,
                        raw_question=question or "",
                        sanitized_retrieval_query=retrieval_query,
                        event_type="contradiction_detection",
                        output_checks={
                            "pairs": len(contradiction_pairs),
                            "topics": [p.topic for p in contradiction_pairs],
                        },
                        auth_context=auth_context,
                        accessed_chunks=sorted(
                            {p.chunk_a_id for p in contradiction_pairs}
                            | {p.chunk_b_id for p in contradiction_pairs}
                        ),
                        llm_version=self._llm_version,
                        embedding_version=self._embedding_version,
                        system_prompt_version=self._system_prompt_version,
                    )
                    if (
                        self._contradiction_db_path is not None
                        and self._contradiction_storage is not None
                    ):
                        self._contradiction_storage(
                            self._contradiction_db_path,
                            trace_id,
                            retrieval_query,
                            contradiction_pairs,
                            retrieved,
                        )
            except Exception as exc:
                logger.warning(
                    "contradiction_detection_failed",
                    extra={"trace_id": trace_id, "error": str(exc)},
                )

        # 5) Cache check. Only the LLM call is skippable -- the key depends on the
        # retrieved chunk ids (in retrieval order, never sorted -- see rag/cache.py), so
        # retrieval must already have run, and it always runs regardless of cache state.
        cache_key: str | None = None
        text: str | None = None
        # Only built below on a cache miss -- stays None on a cache hit (the
        # replay snapshot then just has no verbatim prompt text for that request, which
        # is fine, since replay reconstructs the prompt fresh via service.ask() anyway).
        user_prompt: str | None = None
        if self._cache is not None:
            with instrumented_step("cache_lookup") as rec:
                cache_key = compute_cache_key(
                    retrieval_query,
                    [c.chunk.id for c in retrieved],
                    self._system_prompt_version,
                    self._llm_version,
                    self._embedding_version,
                )
                text = self._cache.get(cache_key)
                cache_hit = text is not None
                rec.set(cache_hit=cache_hit)
            if cache_hit:
                logger.debug("cache hit", extra={"status": "cache_hit"})
                CACHE_HITS.add(1)
            else:
                CACHE_MISSES.add(1)

        # 6) Generate a grounded answer (skipped on a cache hit).
        if text is None:
            with instrumented_step("generate"):
                user_prompt = build_user_prompt(retrieval_query, retrieved, self._messages)
                try:
                    text = self._llm.generate(self._system_prompt, user_prompt)
                except LLMError as exc:
                    raise LLMUnavailableError(str(exc)) from exc
            if cache_key is not None:
                assert self._cache is not None  # cache_key is only set when cache exists
                self._cache.put(cache_key, text)

        # 7) Map the [S#] tags the model actually used back to their sources, and flag
        # any that don't match a real retrieved source. Runs unconditionally
        # (hit or miss) against the *current* retrieved list, never a cached one.
        with instrumented_step("cite") as rec:
            citations, invalid_indices = self._extract_citations(text, retrieved)
            rec.set(citation_count=len(citations))

        if invalid_indices:
            text = self._flag_unverified_citations(text, invalid_indices)

        # 8) SECURITY (v2, opt-in): output-side validation -- citation grounding
        # (annotate only), canary/extraction detection, PII egress (both block, replacing
        # the answer with a safe fallback). Only reached on a pass/restricted v2 decision
        # -- a reject already returned above before the LLM was ever called.
        abstained = False
        if self._guard is not None and guard_decision is not None:
            with instrumented_step("output_guard") as rec:
                outcome = evaluate_output(
                    text,
                    [c.tag for c in citations],
                    [rc.chunk.text for rc in retrieved],
                    canary=CANARY_TOKEN,
                    citation_grounding_threshold=self._guard.citation_grounding_threshold,
                )
                rec.set(
                    canary_detected=outcome.canary_detected,
                    pii_egress_detected=outcome.pii_egress_detected,
                )
            text = outcome.text
            if outcome.canary_detected:
                RAG_CANARY_DETECTED.add(1)
            if outcome.pii_egress_detected:
                RAG_PII_EGRESS.add(1)
            for _ in outcome.ungrounded_citations:
                RAG_UNGROUNDED_CITATIONS.add(1)

            if outcome.blocked:
                citations, retrieved, abstained = [], [], True
            elif guard_decision.zone == "restricted":
                text = f"{text}\n\n{RESTRICTED_WARNING}"

            self._write_audit(
                trace_id=trace_id,
                raw_question=question or "",
                sanitized_retrieval_query=retrieval_query,
                event_type="ok",
                decision=guard_decision,
                guard_version=self._guard.guard_version,
                model_version=self._guard.model_version,
                thresholds=(self._guard.low_threshold, self._guard.high_threshold),
                output_checks={
                    "invalid_citations": [f"S{i}" for i in invalid_indices],
                    "ungrounded_citations": outcome.ungrounded_citations,
                    "canary_detected": outcome.canary_detected,
                    "pii_egress_detected": outcome.pii_egress_detected,
                },
                auth_context=auth_context,
                accessed_chunks=[c.chunk.id for c in retrieved],
            )
        elif self._guard is None:
            # Unconditional per-request audit record when guard v2 is disabled
            # -- the branch above already wrote one when guard ran, so this is the only
            # remaining path that reaches "the end" of _answer() with none written yet.
            self._write_audit(
                trace_id=trace_id,
                raw_question=question or "",
                sanitized_retrieval_query=retrieval_query,
                event_type="ok",
                auth_context=auth_context,
                accessed_chunks=[c.chunk.id for c in retrieved],
            )

        REQUEST_COUNT.add(1, {"status": "ok"})
        return Answer(
            text=text,
            citations=citations,
            retrieved=retrieved,
            redactions=redaction_labels,
            trace_id=trace_id,
            abstained=abstained,
            # Populated only on this, the one exit point that reaches "the end" of
            # _answer() -- the earlier early-return Answers (empty question, legacy-guard
            # block, empty KB, no chunks, low confidence, guard v2 reject) stay None,
            # mirroring the audit records' own "one exit point" scope boundary.
            llm_version=self._llm_version,
            embedding_version=self._embedding_version,
            system_prompt_version=self._system_prompt_version,
            # For the replay snapshot -- same "one exit point" scope as the three
            # fields above.
            retrieval_query=retrieval_query,
            user_prompt=user_prompt,
            # Set above (Step 4.5), before generation even runs -- None whenever
            # detection is disabled, found nothing, or degraded silently on failure.
            contradiction_warning=contradiction_warning,
        )

    @staticmethod
    def _extract_citations(
        text: str, retrieved: list[RetrievedChunk]
    ) -> tuple[list[Citation], list[int]]:
        """Map ``[S#]`` tags the model used back to their sources.

        Returns ``(citations, invalid_indices)``: ``invalid_indices`` are the tag numbers
        that do not correspond to any retrieved source -- previously these were
        silently dropped with no signal to the caller; now the caller decides what to do
        with them (see ``_flag_unverified_citations``).
        """

        used = sorted({int(m) for m in _TAG_RE.findall(text)})
        citations: list[Citation] = []
        invalid_indices: list[int] = []
        for idx in used:
            if 1 <= idx <= len(retrieved):
                item = retrieved[idx - 1]
                c = item.chunk
                citations.append(
                    Citation(
                        tag=f"S{idx}",
                        document_id=c.document_id,
                        title=c.title,
                        doc_type=c.doc_type,
                        source_path=c.source_path,
                        score=item.score,
                    )
                )
            else:
                invalid_indices.append(idx)
        return citations, invalid_indices

    def _flag_unverified_citations(self, text: str, invalid_indices: list[int]) -> str:
        """Append a warning for ``[S#]`` tags that cite no real retrieved source.

        Verification must never block the response, so this is wrapped in its own
        try/except -- a logging or metrics failure here must not turn an otherwise
        successful answer into a failed request (same idiom as
        ``llm/token_tracking.py``'s ``_record_usage``).
        """

        try:
            tags = [f"[S{i}]" for i in invalid_indices]
            logger.warning(
                "response cited unverifiable sources",
                extra={"status": "unverified_citations", "invalid_citations": tags},
            )
            for idx in invalid_indices:
                record_unverified_citation(idx)
            joined = ", ".join(f"S{i}" for i in invalid_indices)
            return text + self._messages.unverified_citations_warning.format(joined=joined)
        except Exception:
            # Citation verification is best-effort; failures must not affect the returned answer.
            logger.exception("citation verification failed (answer unaffected)")
            return text
