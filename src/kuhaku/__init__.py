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
import logging
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .core.auth import AuthContext
from .core.config import Settings, get_settings
from .core.exceptions import SecurityComponentError
from .core.llm import build_llm_provider
from .core.llm.token_tracking import TokenTrackingLLM
from .core.models import ExecutionResult, Message, ToolCall
from .core.policy import enforce_guard_policy, validate_audit_log_path
from .core.sanitization import Redaction
from .tools.rag import (
    ACCESS_TAG_INTERNAL,
    ACCESS_TAG_PUBLIC,
    ACCESS_TAG_RESTRICTED,
    Answer,
    ChromaVectorStore,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    QueryAnswerCache,
    RAGEngine,
    RAGSettings,
    Retriever,
    SparseRetriever,
    UnsupportedFileType,
    build_bm25_from_store,
    build_chunker,
    build_embedding_provider,
    extract_text,
)
from .tools.rag.prompts import load_system_prompt

logger = logging.getLogger(__name__)

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
    includes them.

    Document-level access filtering (:meth:`ingest`'s/:meth:`load_documents`'s
    ``access_tags``, :meth:`ask`'s ``auth_context``) is available directly on this
    facade -- see each method. ``kuhaku`` itself authenticates no one: the caller
    authenticates its own user and hands this an :class:`~kuhaku.core.auth.AuthContext`
    (identity + roles); a chunk tagged with ``access_tags`` is retrievable only when at
    least one tag is also in that context's ``roles``.

    For anything else not exposed here (query rewriting, contradiction detection,
    caching, the security guard, custom ``EngineMessages``, the prompt's format-
    preference/example/masked-placeholder layers, ...) construct :class:`RAGEngine`
    directly -- see ``kuhaku.tools.rag`` -- or use the :attr:`engine` escape hatch below.
    """

    def __init__(
        self,
        *,
        retrieval: str | None = None,
        reranker: bool | str | None = None,
        chunking: str | None = None,
        embedding: str | None = None,
        vector_store: str | None = None,
        corpus_dir: str | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        audit_enabled: bool | None = None,
        audit_log_path: str | None = None,
        cache: bool | str | None = None,
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
            retrieval: ``"dense"`` (embedding similarity), ``"sparse"`` (BM25 keyword
                search, built from whatever is already in the vector store), or
                ``"hybrid"`` (both, fused with RRF); ``None`` (the default) uses
                ``RAGSettings.retrieval``, which is ``"hybrid"`` by default -- a caller
                who configures nothing gets hybrid retrieval. Hybrid costs CPU/memory
                only (no model download), and covers dense's weak spot on exact-match/
                rare-term queries via BM25. Neither ``"sparse"`` nor ``"hybrid"`` needs
                ``corpus_dir``: BM25 is built from the store's own chunks, not by
                re-reading source files from disk, and it picks up documents ingested
                after construction via :meth:`ingest`. That BM25 index is built lazily,
                on the first query after construction (or after an ingest) -- not at
                construction time -- so a caller with a large corpus should expect that
                first query to pay an O(corpus size) indexing cost; ``retrieval="dense"``
                avoids it entirely.
            reranker: ``None`` (the default) defers to ``RAGSettings.rerank_enabled``
                (``False`` by default -- ``reranker_model`` is roughly a gigabyte to
                download plus VRAM, so a bare ``RAG()`` never fetches it; set
                ``KUHAKU_RAG__RERANK_ENABLED=true`` to opt in without touching code).
                ``False`` forces re-ranking off outright, overriding
                ``rerank_enabled``. ``True`` forces it on using
                ``RAGSettings.reranker_model``. A non-empty ``str`` forces it on using
                that HuggingFace cross-encoder model name instead. An empty string
                (``""``) is treated the same as ``False`` -- off, not a request to load
                a blank model name.
            chunking: ``"paragraph"`` or ``"structural"``; ``None`` uses
                ``RAGSettings.chunking_strategy``.
            embedding: an embedding model name; ``None`` uses
                ``RAGSettings.embedding_model``.
            vector_store: a Chroma persistence directory; ``None`` uses
                ``RAGSettings.chroma_persist_dir`` if set, otherwise a fresh temporary
                directory (so a bare ``RAG()`` never depends on -- or writes into --
                the current working directory).
            corpus_dir: accepted for ``RAGSettings`` compatibility, but not read by
                anything ``RAG`` does -- neither construction nor any retrieval mode
                loads documents from it; use :meth:`load_documents` to ingest a
                directory instead. ``None`` uses ``RAGSettings.corpus_dir``.
            vertex_project: Google Cloud project for the ``vertex`` LLM/embedding
                providers; ``None`` uses ``Settings.vertex_project``.
            vertex_location: Google Cloud location for the ``vertex`` LLM/embedding
                providers; ``None`` uses ``Settings.vertex_location``.
            audit_enabled: whether audit logging is on; ``None`` uses
                ``Settings.audit_enabled`` (default ``True``).
            audit_log_path: where audit records are written; ``None`` uses
                ``Settings.audit_log_path`` if set, otherwise the kuhaku-managed
                default (``./logs/kuhaku_audit.jsonl``).
            cache: the query-answer cache. ``None`` (default) defers to
                ``RAGSettings.cache_enabled`` (``True`` by default) and
                ``RAGSettings.cache_db_path`` -- a caller who configures nothing gets
                caching, at the kuhaku-managed default location
                (``./data/kuhaku_qa_cache.sqlite3``). ``False`` disables caching
                outright (no cache file is created). ``True`` forces caching on at
                ``RAGSettings.cache_db_path``/the default location. A ``str`` forces
                caching on at that explicit path, overriding ``RAGSettings.cache_db_path``.
                ``RAGSettings.cache_ttl_seconds`` governs expiry in every case where a
                cache is built. Building a cache here never itself touches disk --
                schema creation is lazy, on the first request that actually reaches the
                cache lookup (see ``rag/cache.py``) -- so a bare ``RAG()`` (or one whose
                every request abstains before that point) creates no file. A cache that
                cannot be created or opened when it is finally used (read-only
                directory, corrupt file) degrades to no caching with a logged warning --
                it never fails a query.
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
            **kwargs: any field name from :class:`Settings` or
                :class:`~kuhaku.tools.rag.config.RAGSettings` that has no dedicated
                parameter above (e.g. ``guard_enabled``, ``retry_llm_max_attempts``,
                ``chunk_size``) is accepted here and applied as an override on ``settings``/
                ``rag_settings`` respectively -- equivalent to setting it on a
                :class:`Settings`/:class:`RAGSettings` instance yourself, just without
                needing to build one. A name matching neither still raises ``TypeError``,
                so a typo is caught exactly as before this existed. This means a new
                field added to either settings class becomes usable here immediately,
                with no change to this constructor.
        """

        # Anything left in kwargs after the named convenience parameters above have
        # already claimed their names is either a typo, or the name of a field that
        # lives on Settings/RAGSettings but has no dedicated convenience parameter here
        # (`guard_enabled` is the running example -- see AGENTS.md's known gaps). Rather
        # than reject the latter, route it generically: any name found on
        # Settings.model_fields becomes a Settings override, any name found on
        # RAGSettings' dataclass fields becomes a RAGSettings override (Settings wins
        # the three names both classes share -- retry_enabled, vertex_project,
        # vertex_location -- since RAGSettings.from_settings() already derives its own
        # copy from Settings, so overriding Settings alone is enough to reach both).
        # Only a name matching neither is still a hard TypeError, so a genuine typo
        # (`RAG(guard_enalbed=True)`) is caught exactly as before -- this widens what a
        # valid kwarg name can be, it does not loosen the "unknown names fail loudly"
        # guarantee. A name already claimed by one of the explicit parameters above
        # (`retrieval`, `audit_enabled`, ...) can never reach here at all: Python binds
        # it to that parameter first, so there is no dual-path way to set the same knob.
        settings_field_names = set(Settings.model_fields) - {"rag"}
        rag_settings_field_names = {f.name for f in dataclasses.fields(RAGSettings)}
        extra_settings_kwargs: dict[str, object] = {}
        extra_rag_kwargs: dict[str, object] = {}
        for key in list(kwargs):
            if key in settings_field_names:
                extra_settings_kwargs[key] = kwargs.pop(key)
            elif key in rag_settings_field_names:
                extra_rag_kwargs[key] = kwargs.pop(key)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"RAG.__init__() got an unexpected keyword argument {unexpected!r}")

        base = settings or get_settings()
        settings_overrides: dict[str, object] = dict(extra_settings_kwargs)
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
        rag_overrides: dict[str, object] = dict(extra_rag_kwargs)
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

        # One place decides the effective retrieval strategy: an explicit `retrieval`
        # argument always wins; `None` defers to `RAGSettings.retrieval` (default
        # "hybrid"). Whichever value wins is validated the same way regardless of
        # source, so a bad KUHAKU_RAG__RETRIEVAL value fails exactly like a bad
        # explicit argument -- no silent fallback to a strategy nobody asked for.
        effective_retrieval = retrieval if retrieval is not None else self._rag_settings.retrieval
        if effective_retrieval not in _VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval must be one of {_VALID_RETRIEVAL_MODES}; got {effective_retrieval!r}"
            )

        # Fail fast, at construction, before building anything (the embedder may
        # download a model) rather than lazily discovering these at the first ask()/
        # ingest() call. Two different severities, matching kuhaku.core.policy's
        # three-tier design: `guard_enabled=True` is something the caller explicitly
        # opted into through this very constructor's Settings, and RAG() does not build
        # a GuardPipeline for it -- that broken promise must not pass silently, so it
        # raises. The audit log is on (`audit_enabled=True`) by default for everyone,
        # opted into by nobody explicitly -- unwritable, it is reported (with the
        # reason) as a warning and construction continues, same as every other
        # performance/helper component's fallback policy.
        try:
            enforce_guard_policy(self._settings, {"guard": None})
        except SecurityComponentError as exc:
            raise SecurityComponentError(
                f"{exc} RAG() does not build a GuardPipeline itself yet -- construct "
                "one and call rag.engine.update_guard(guard), or set "
                "guard_enabled=False to opt out."
            ) from exc
        try:
            validate_audit_log_path(self._settings)
        except SecurityComponentError as exc:
            logger.warning(str(exc))

        s = self._settings
        rs = self._rag_settings
        self._chunker = build_chunker(rs)
        self._embedder = build_embedding_provider(rs)
        self._store = ChromaVectorStore(
            rs.chroma_persist_dir,
            rs.chroma_collection,
            retry_enabled=rs.retry_enabled,
            retry_max_attempts=rs.retry_vectorstore_max_attempts,
            retry_backoff_seconds=rs.retry_vectorstore_backoff_seconds,
        )
        llm = build_llm_provider(s)
        if enable_token_tracking and not isinstance(llm, TokenTrackingLLM):
            llm = TokenTrackingLLM(llm, provider=s.llm_provider.strip().lower())
        self._llm = llm
        self._retriever = self._build_retriever(effective_retrieval, reranker)
        self._cache = self._build_cache(cache)

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
            cache=self._cache,
            system_prompt=resolved_system_prompt,
            rag_settings=rs,
        )

    def _build_retriever(self, retrieval: str, reranker: bool | str | None) -> Retriever:
        rs = self._rag_settings
        dense = DenseRetriever(self._embedder, self._store)

        sparse = None
        if retrieval in ("sparse", "hybrid"):
            # Sourced from the same store the dense side reads -- not from corpus_dir --
            # so both sides always chunk each document exactly once (Feature 2).
            sparse = build_bm25_from_store(self._store, k1=rs.bm25_k1, b=rs.bm25_b)

        # One place decides the effective re-ranker: an explicit `reranker` (including
        # `False`/`""`, both "off") always wins; `None` defers to
        # `RAGSettings.rerank_enabled` (`False` by default -- see config.py -- so a bare
        # RAG() builds no CrossEncoderReranker and downloads nothing).
        if reranker is None:
            effective_reranker: bool | str = rs.reranker_model if rs.rerank_enabled else False
        else:
            effective_reranker = reranker

        reranker_instance = None
        if effective_reranker:
            model_name = (
                effective_reranker if isinstance(effective_reranker, str) else rs.reranker_model
            )
            reranker_instance = CrossEncoderReranker(
                model_name,
                device=rs.reranker_device,
                retry_enabled=rs.retry_enabled,
                retry_max_attempts=rs.retry_reranker_max_attempts,
                retry_backoff_base_seconds=rs.retry_reranker_backoff_base_seconds,
                retry_backoff_max_seconds=rs.retry_reranker_backoff_max_seconds,
            )

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

    def _build_cache(self, cache: bool | str | None) -> QueryAnswerCache | None:
        """The one construction site for the query-answer cache.

        Three-state (``bool | str | None``), same convention :meth:`_build_retriever`'s
        ``reranker`` uses for ``RAGSettings.rerank_enabled``: ``None`` means "the caller
        configured nothing", distinct from an explicit ``False``, so it can defer to
        ``RAGSettings.cache_enabled``/``cache_db_path`` rather than forcing a
        facade-level default that would silently override them -- mirroring
        ``audit_enabled: bool | None``'s own precedent above.

        Never raises: ``QueryAnswerCache.__init__`` does no I/O at all (schema creation
        is lazy -- see ``rag/cache.py``), and this adds one more layer of defense so a
        cache that fails to build for some other reason disables caching rather than
        failing ``RAG()`` construction outright.
        """

        rs = self._rag_settings
        if cache is False:
            return None
        if cache is None:
            if not rs.cache_enabled:
                return None
            db_path = rs.cache_db_path
        elif cache is True:
            db_path = rs.cache_db_path
        else:
            db_path = cache

        try:
            return QueryAnswerCache(db_path, rs.cache_ttl_seconds)
        except Exception:
            logger.warning(
                "failed to initialize the query-answer cache; caching disabled",
                exc_info=True,
            )
            return None

    def ingest(
        self,
        text: str,
        filename: str = "document.txt",
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
        access_tags: Sequence[str] | None = None,
    ) -> tuple[int, list[Redaction]]:
        """Sanitize, chunk, embed, and add one document to the vector store.

        Thin pass-through to :meth:`RAGEngine.ingest_document` -- the document is
        visible to the very next :meth:`ask` call, no reload needed. Returns
        ``(chunks_added, redactions)``. ``chunk_size``/``overlap`` default to
        ``RAGSettings.chunk_size``/``RAGSettings.chunk_overlap`` when not given.

        ``access_tags``, when given, restricts every chunk this document produces to
        callers whose :meth:`ask` ``auth_context.roles`` includes at least one of these
        tags (document-level access filtering) -- ``kuhaku`` itself never interprets what
        a tag means. ``None`` (the default) leaves the document untagged: visible to
        every caller, exactly as before this parameter existed. A blank tag raises
        ``ValueError`` rather than being silently stored as something that can never
        match.
        """

        rs = self._rag_settings
        return self._engine.ingest_document(
            text,
            filename,
            chunk_size=chunk_size if chunk_size is not None else rs.chunk_size,
            overlap=overlap if overlap is not None else rs.chunk_overlap,
            access_tags=access_tags,
        )

    def load_documents(
        self, directory: str | Path, *, access_tags: Sequence[str] | None = None
    ) -> int:
        """Ingest every supported file (``.txt``, ``.md``, ``.pdf``) in ``directory``.

        A thin loop over :meth:`ingest` -- no indexing logic is duplicated here.
        Non-recursive (top-level files only). Prints one ``indexed <filename>: N
        chunk(s)`` line per file and returns the number of documents indexed. A file
        that can't be read (bad encoding, unreadable PDF) is skipped with a warning
        rather than aborting the whole batch.

        ``access_tags``, when given, is applied uniformly to every file this call
        indexes -- per-file tagging is deliberately not supported here; call
        :meth:`ingest` directly for that. ``None`` (the default) leaves every file
        untagged.
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

            chunks_added, _redactions = self.ingest(
                text, file_path.name, access_tags=access_tags
            )
            print(f"indexed {file_path.name}: {chunks_added} chunk(s)")
            indexed += 1

        return indexed

    def ask(
        self,
        question: str,
        context_text: str | None = None,
        *,
        auth_context: AuthContext | None = None,
    ) -> Answer:
        """Answer ``question``, optionally grounded by supplementary structured context.

        ``context_text`` is an optional blob (JSON, XML, or raw text) supplied alongside
        the question -- e.g. something a user pasted or uploaded -- whose salient fields
        are extracted and folded into the retrieval query.

        ``auth_context`` (see :class:`~kuhaku.core.auth.AuthContext`) is the caller's
        already-authenticated identity -- ``kuhaku`` never authenticates anyone itself.
        Retrieval enforces document-level access filtering against it unconditionally: a
        chunk tagged via :meth:`ingest`'s/:meth:`load_documents`' ``access_tags`` is
        retrievable only when at least one tag is also in ``auth_context.roles``; an
        untagged chunk is retrievable by anyone. ``None`` (the default) is treated as
        having no roles, so only untagged chunks are visible -- existing callers that
        pass nothing keep working unchanged against an untagged corpus.

        Thin pass-through to :meth:`RAGEngine.answer` -- see there for the full
        contract (sanitization, citations, abstention, ...).
        """

        return self._engine.answer(question, context_text, auth_context=auth_context)

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
    # Document-level access filtering: the identity primitive RAG.ask()'s auth_context
    # expects, plus a default tag vocabulary RAG.ingest()'s access_tags can reach for
    # (Feature 6, suggestions only -- any other string works identically).
    "AuthContext",
    "ACCESS_TAG_PUBLIC",
    "ACCESS_TAG_INTERNAL",
    "ACCESS_TAG_RESTRICTED",
]
