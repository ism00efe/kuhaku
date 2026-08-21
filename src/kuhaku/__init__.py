"""kuhaku — generic, tool-agnostic runtime infrastructure (``kuhaku.core``), a
generic evaluation harness (``kuhaku.evaluation``), and RAG as one tool built on top
of both (``kuhaku.tools.rag``).

This top-level module re-exports two things: the generic core primitives
(:class:`Settings`, :func:`build_llm_provider`, :class:`Message`/:class:`ToolCall`/
:class:`ExecutionResult`) any tool built on kuhaku can use, and :class:`RAG` --
a thin, opinionated wrapper around :class:`~kuhaku.tools.rag.engine.RAGEngine` for
callers who don't need the full composition-root control the individual RAG building
blocks offer (see ``kuhaku.tools.rag`` for those). Every knob :class:`RAG` takes is
optional -- anything left unset falls back to :class:`~kuhaku.core.config.Settings`'
own default/env-var value, the same convention every other part of kuhaku
follows.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path
from typing import Any

from .core.config import Settings, get_settings
from .core.llm import build_llm_provider
from .core.llm.token_tracking import TokenTrackingLLM
from .core.models import ExecutionResult, Message, ToolCall
from .core.sanitization import Redaction
from .tools.rag import (
    Answer,
    ChromaVectorStore,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    RAGEngine,
    RAGSettings,
    Retriever,
    SparseRetriever,
    UnsupportedFileType,
    build_bm25_from_corpus,
    build_chunker,
    build_embedding_provider,
    extract_text,
)
from .tools.rag.prompts import load_system_prompt

__version__ = "0.1.0"

_VALID_RETRIEVAL_MODES = ("dense", "sparse", "hybrid")
_LOAD_DOCUMENTS_SUFFIXES = (".txt", ".md", ".pdf")


class RAG:
    """Retrieval-augmented generation: sanitize -> retrieve -> generate, with citations.

    Wraps :class:`RAGEngine` behind a handful of simple, named parameters instead of its
    full constructor-injection surface. The system prompt is one of those parameters:
    ``persona``/``language_policy`` each override one caller-owned layer of the
    framework's default, layered prompt (see
    :func:`~kuhaku.tools.rag.prompts.load_system_prompt`), while ``system_prompt``
    replaces it outright. Replacing it outright means taking full ownership of the
    safety rules -- kuhaku's instruction-precedence, data-marking, canary, grounding,
    citation, and contradiction-handling rules only apply if your replacement text
    includes them. For anything else not exposed here (query rewriting, contradiction
    detection, caching, the security guard, authentication/authorization via
    ``kuhaku.core.auth``, custom ``EngineMessages``, the prompt's format-preference/
    example/masked-placeholder layers, ...) construct :class:`RAGEngine` directly -- see
    ``kuhaku.tools.rag`` -- or use the :attr:`engine` escape hatch below (e.g.
    ``rag.engine.update_authorization_policy(...)`` and passing ``auth_context=`` to
    ``rag.engine.answer(...)``).
    """

    def __init__(
        self,
        *,
        retrieval: str = "dense",
        reranker: bool | str = False,
        chunking: str | None = None,
        embedding: str | None = None,
        vector_store: str | None = None,
        corpus_dir: str | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        audit_enabled: bool | None = None,
        audit_log_path: str | None = None,
        persona: str | None = None,
        language_policy: str | None = None,
        system_prompt: str | None = None,
        settings: Settings | None = None,
        rag_settings: RAGSettings | None = None,
        enable_token_tracking: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            retrieval: ``"dense"`` (embedding similarity, the default), ``"sparse"``
                (BM25 keyword search, requires ``corpus_dir``), or ``"hybrid"`` (both,
                fused with RRF, requires ``corpus_dir``).
            reranker: ``False`` (default, no re-ranking), ``True`` (cross-encoder
                re-ranking using ``RAGSettings.reranker_model``), or a specific
                HuggingFace cross-encoder model name.
            chunking: ``"paragraph"`` or ``"structural"``; ``None`` uses
                ``RAGSettings.chunking_strategy``.
            embedding: an embedding model name; ``None`` uses
                ``RAGSettings.embedding_model``.
            vector_store: a Chroma persistence directory; ``None`` uses
                ``RAGSettings.chroma_persist_dir`` if set, otherwise a fresh temporary
                directory (so a bare ``RAG()`` never depends on -- or writes into --
                the current working directory).
            corpus_dir: directory of source documents, only needed for
                ``retrieval="sparse"``/``"hybrid"`` (BM25 is rebuilt from disk, not
                stored); ``None`` uses ``RAGSettings.corpus_dir``.
            vertex_project: Google Cloud project for the ``vertex`` LLM/embedding
                providers; ``None`` uses ``Settings.vertex_project``.
            vertex_location: Google Cloud location for the ``vertex`` LLM/embedding
                providers; ``None`` uses ``Settings.vertex_location``.
            audit_enabled: whether audit logging is on; ``None`` uses
                ``Settings.audit_enabled`` (default ``True``).
            audit_log_path: where audit records are written; ``None`` uses
                ``Settings.audit_log_path`` if set, otherwise the kuhaku-managed
                default (``./logs/kuhaku_audit.jsonl``).
            persona: overrides the system prompt's persona layer (who the assistant is);
                ``None`` uses the framework's neutral default (a general-purpose
                assistant that answers from the supplied reference material). One layer
                of :func:`~kuhaku.tools.rag.prompts.load_system_prompt` -- the
                safety/grounding core (instruction precedence, data marking, the canary
                rule, grounding, citations, contradiction handling, masked-value
                preservation) is not affected by this and cannot be dropped through it.
            language_policy: overrides the system prompt's output-language-policy layer;
                ``None`` uses the framework default (answer in the language of the
                question). Same safety-core guarantee as ``persona`` above.
            system_prompt: replaces the entire system prompt outright, bypassing
                ``persona``/``language_policy`` (and every other layer) entirely; ``None``
                (the default) uses the layered, framework-assembled prompt. Supplying
                this means taking full ownership of the safety rules yourself --
                kuhaku no longer enforces instruction precedence, data marking, the
                canary rule, grounding, citations, or any of the other rules
                :func:`~kuhaku.tools.rag.prompts.load_system_prompt`'s template provides
                unless your replacement text includes them.
            settings: a pre-built :class:`Settings` instance for the generic, cross-tool
                knobs (LLM provider, Vertex auth, audit logging); ``None`` uses
                :func:`get_settings`.
            rag_settings: a pre-built :class:`~kuhaku.tools.rag.config.RAGSettings`
                instance for every RAG-specific knob; ``None`` (the default) derives one
                from ``settings`` (:meth:`RAGSettings.from_settings`). When given
                explicitly, it is used as-is -- the ``chunking``/``embedding``/
                ``vector_store``/``corpus_dir`` convenience kwargs above still apply as
                overrides on top of it, but it is not itself derived from ``settings``.
            enable_token_tracking: whether to wrap the LLM provider with
                :class:`~kuhaku.core.llm.token_tracking.TokenTrackingLLM`, which
                records per-call token usage (Prometheus metrics + a log line). ``True``
                by default; pass ``False`` to skip it.
        """

        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"RAG.__init__() got an unexpected keyword argument {unexpected!r}")

        if retrieval not in _VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval must be one of {_VALID_RETRIEVAL_MODES}; got {retrieval!r}"
            )

        base = settings or get_settings()
        settings_overrides: dict[str, object] = {}
        if vertex_project is not None:
            settings_overrides["vertex_project"] = vertex_project
        if vertex_location is not None:
            settings_overrides["vertex_location"] = vertex_location
        if audit_enabled is not None:
            settings_overrides["audit_enabled"] = audit_enabled
        if audit_log_path is not None:
            settings_overrides["audit_log_path"] = audit_log_path
        self._settings = base.model_copy(update=settings_overrides) if settings_overrides else base

        # Feature 1 (kuhaku tech-debt cleanup): every RAG-scoped knob (chunking, the
        # vector store, retrieval/re-ranking) is sourced from this RAGSettings instance,
        # never from the flat Settings object -- see kuhaku.tools.rag.config.
        # RAGSettings. Fields RAGSettings doesn't carry (LLM provider selection, Vertex
        # auth, audit logging) keep reading self._settings directly below, since they're
        # cross-tool concerns.
        base_rag = rag_settings if rag_settings is not None else RAGSettings.from_settings(
            self._settings
        )
        rag_overrides: dict[str, object] = {}
        if chunking is not None:
            rag_overrides["chunking_strategy"] = chunking
        if embedding is not None:
            rag_overrides["embedding_model"] = embedding
        if vector_store is not None:
            rag_overrides["chroma_persist_dir"] = vector_store
        elif not base_rag.chroma_persist_dir:
            rag_overrides["chroma_persist_dir"] = tempfile.mkdtemp(prefix="kuhaku_rag_")
        if corpus_dir is not None:
            rag_overrides["corpus_dir"] = corpus_dir
        self._rag_settings = (
            dataclasses.replace(base_rag, **rag_overrides) if rag_overrides else base_rag
        )

        s = self._settings
        rs = self._rag_settings
        self._chunker = build_chunker(rs)
        self._embedder = build_embedding_provider(rs)
        self._store = ChromaVectorStore(rs.chroma_persist_dir, rs.chroma_collection)
        llm = build_llm_provider(s)
        if enable_token_tracking and not isinstance(llm, TokenTrackingLLM):
            llm = TokenTrackingLLM(llm, provider=s.llm_provider.strip().lower())
        self._llm = llm
        self._retriever = self._build_retriever(retrieval, reranker)

        # `system_prompt` replaces the whole thing outright; `persona`/`language_policy`
        # each override one layer of the framework-assembled prompt while leaving the
        # safety core (and every other layer) at its default -- see load_system_prompt().
        # `None` (nothing given) is passed straight through to RAGEngine, which already
        # falls back to its own module-level default -- no separate "resolved" constant
        # needed here.
        resolved_system_prompt = system_prompt
        if resolved_system_prompt is None and (persona is not None or language_policy is not None):
            layer_overrides: dict[str, str] = {}
            if persona is not None:
                layer_overrides["persona"] = persona
            if language_policy is not None:
                layer_overrides["language_policy"] = language_policy
            resolved_system_prompt = load_system_prompt(**layer_overrides)

        self._engine = RAGEngine(
            self._embedder,
            self._store,
            self._llm,
            top_k=rs.top_k,
            retriever=self._retriever,
            chunker=self._chunker,
            confidence_threshold=rs.rag_confidence_threshold,
            audit_enabled=s.audit_enabled,
            audit_log_path=s.audit_log_path or "",
            system_prompt=resolved_system_prompt,
            rag_settings=rs,
        )

    def _build_retriever(self, retrieval: str, reranker: bool | str) -> Retriever:
        rs = self._rag_settings
        dense = DenseRetriever(self._embedder, self._store)

        sparse = None
        if retrieval in ("sparse", "hybrid"):
            sparse = build_bm25_from_corpus(
                rs.corpus_dir,
                chunk_size=rs.chunk_size,
                overlap=rs.chunk_overlap,
                k1=rs.bm25_k1,
                b=rs.bm25_b,
                chunker=self._chunker,
                rag_settings=rs,
            )

        reranker_instance = None
        if reranker:
            model_name = reranker if isinstance(reranker, str) else rs.reranker_model
            reranker_instance = CrossEncoderReranker(model_name)

        if retrieval == "dense" and reranker_instance is None:
            return dense

        # `retrieval == "sparse"` is its own orchestration path, never `HybridRetriever`
        # (which is reserved for genuine dense+sparse fusion): `SparseRetriever` is what
        # normalizes BM25's raw, unbounded Okapi scores into the 0..1 range the
        # confidence threshold expects (see retriever.py) -- returning the bare
        # `BM25Retriever` here would skip that.
        if retrieval == "sparse":
            return SparseRetriever(
                sparse,
                reranker_instance,
                candidates=rs.rerank_candidates,
                max_chunks_per_document=rs.max_chunks_per_document,
            )

        return HybridRetriever(
            dense,
            sparse,
            reranker_instance,
            rrf_k=rs.rrf_k,
            candidates=rs.rerank_candidates,
            max_chunks_per_document=rs.max_chunks_per_document,
        )

    def ingest(
        self,
        text: str,
        filename: str = "document.txt",
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> tuple[int, list[Redaction]]:
        """Sanitize, chunk, embed, and add one document to the vector store.

        Thin pass-through to :meth:`RAGEngine.ingest_document` -- the document is
        visible to the very next :meth:`ask` call, no reload needed. Returns
        ``(chunks_added, redactions)``. ``chunk_size``/``overlap`` default to
        ``RAGSettings.chunk_size``/``RAGSettings.chunk_overlap`` when not given.
        """

        rs = self._rag_settings
        return self._engine.ingest_document(
            text,
            filename,
            chunk_size=chunk_size if chunk_size is not None else rs.chunk_size,
            overlap=overlap if overlap is not None else rs.chunk_overlap,
        )

    def load_documents(self, directory: str | Path) -> int:
        """Ingest every supported file (``.txt``, ``.md``, ``.pdf``) in ``directory``.

        A thin loop over :meth:`ingest` -- no indexing logic is duplicated here.
        Non-recursive (top-level files only). Prints one ``indexed <filename>: N
        chunk(s)`` line per file and returns the number of documents indexed. A file
        that can't be read (bad encoding, unreadable PDF) is skipped with a warning
        rather than aborting the whole batch.
        """

        path = Path(directory)
        if not path.is_dir():
            raise FileNotFoundError(f"load_documents: no such directory: {path}")

        files = sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in _LOAD_DOCUMENTS_SUFFIXES
        )
        if not files:
            allowed = ", ".join(_LOAD_DOCUMENTS_SUFFIXES)
            print(f"load_documents: no supported files ({allowed}) found in {path}")
            return 0

        indexed = 0
        for file_path in files:
            try:
                if file_path.suffix.lower() == ".pdf":
                    text = extract_text(file_path.name, file_path.read_bytes())
                else:
                    text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, UnsupportedFileType) as exc:
                print(f"load_documents: skipping {file_path.name}: {exc}")
                continue

            chunks_added, _redactions = self.ingest(text, file_path.name)
            print(f"indexed {file_path.name}: {chunks_added} chunk(s)")
            indexed += 1

        return indexed

    def ask(self, question: str, context_text: str | None = None) -> Answer:
        """Answer ``question``, optionally grounded by supplementary structured context.

        ``context_text`` is an optional blob (JSON, XML, or raw text) supplied alongside
        the question -- e.g. something a user pasted or uploaded -- whose salient fields
        are extracted and folded into the retrieval query.

        Thin pass-through to :meth:`RAGEngine.answer` -- see there for the full
        contract (sanitization, citations, abstention, ...).
        """

        return self._engine.answer(question, context_text)

    def chat_repl(self) -> None:
        """Interactive terminal chat loop: read a question, print the answer, repeat.

        Type ``exit``/``quit`` (case-insensitive) or press Ctrl+C/Ctrl+D to leave.
        Errors raised by :meth:`ask` (e.g. an unavailable LLM provider) are caught and
        printed so one bad turn doesn't end the session.
        """

        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass

        print("Type 'exit' or 'quit' to leave.")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break

            try:
                answer = self.ask(question)
            except Exception as exc:
                print(f"[error] {exc}")
                continue

            print(answer.text)
            if answer.citations:
                print("Sources:")
                for citation in answer.citations:
                    print(f"  - {citation.title} (score={citation.score:.3f})")

    @property
    def engine(self) -> RAGEngine:
        """The underlying :class:`RAGEngine`, for uses this wrapper doesn't expose."""

        return self._engine


__all__ = [
    "__version__",
    "RAG",
    # Generic core primitives, re-exported for callers that build a non-RAG tool on
    # kuhaku without reaching into `kuhaku.core` directly.
    "Settings",
    "get_settings",
    "build_llm_provider",
    "Message",
    "ToolCall",
    "ExecutionResult",
]
