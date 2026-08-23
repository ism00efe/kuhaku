"""RAG-specific domain models.

Plain dataclasses with no external dependencies. These are the vocabulary the RAG
pipeline speaks (documents, chunks, retrieval results, citations, answers). They live
under ``tools/rag`` (not ``kuhaku/core``) because they are specific to the RAG tool --
a different tool built on kuhaku has no reason to know what a ``Chunk`` is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# Document-level access filtering: a small, domain-neutral starting vocabulary a caller
# can reach for instead of inventing their own tag strings (Feature 6). Plain strings and
# nothing more -- no enum, no validation, no implied ordering or hierarchy. Passing any
# other string works identically; these exist purely as a suggestion.
ACCESS_TAG_PUBLIC = "public"
ACCESS_TAG_INTERNAL = "internal"
ACCESS_TAG_RESTRICTED = "restricted"

# Chroma-specific metadata key (see vectorstore.ChromaVectorStore/retriever.py's
# `_entitlement_where`), NOT one of the keys Chunk.metadata() itself returns: Chroma
# rejects an empty-list metadata value outright and has no "key is absent" where-operator
# (verified against the installed client), so an untagged chunk is instead marked with
# this boolean key at write time, letting a `where` filter select "carries no tags"
# without either. Defined here (not in vectorstore.py) purely so vectorstore.py and
# retriever.py -- and tests standing in for either -- agree on the same literal.
ACCESS_TAGS_NONE_KEY = "access_tags_none"


@dataclass(frozen=True)
class Document:
    """A source document in the knowledge base (synthetic or, later, real)."""

    id: str
    title: str
    doc_type: str
    text: str
    source_path: str
    metadata: dict[str, str] = field(default_factory=dict)
    # Freshness metadata, parsed from the document's prose Metadata line by
    # rag/ingestion.py. "" means "no bound" on that side -- the overwhelming majority
    # of documents author none of this and are always fresh, unchanged from before.
    effective_date: str = ""
    obsolete: bool = False
    expiry_date: str = ""
    # Document-level access filtering: which access tags, if any, gate this document.
    # Empty (the default) means "visible to everyone" -- tagging is opt-in. Propagated
    # onto every Chunk this document produces at chunk time (see chunking.py), the same
    # way effective_date/obsolete/expiry_date already are.
    access_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a :class:`Document`."""

    id: str
    document_id: str
    title: str
    doc_type: str
    text: str
    chunk_index: int
    source_path: str
    content_type: str = "text"  # "text" | "table" | "glossary" -- see rag/chunking.py
    # Freshness metadata, inherited from the parent Document at chunk time. Kept as
    # plain str/bool (never None) because ChromaDB metadata values must be str/int/
    # float/bool.
    effective_date: str = ""
    obsolete: bool = False
    expiry_date: str = ""
    # Same access-filtering field as Document, inherited at chunk time (chunking.py).
    # Flat set semantics only: a chunk is visible when untagged, or when at least one tag
    # here also appears in an AuthContext's roles (see retriever.py's `is_entitled`, the
    # single place this field's meaning is interpreted) -- no hierarchy, no ordering.
    access_tags: tuple[str, ...] = ()

    def metadata(self) -> dict[str, str | int | bool | list[str]]:
        """Flat metadata dict stored alongside the vector (used for citations)."""

        return {
            "document_id": self.document_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "chunk_index": self.chunk_index,
            "source_path": self.source_path,
            "content_type": self.content_type,
            "effective_date": self.effective_date,
            "obsolete": self.obsolete,
            "expiry_date": self.expiry_date,
            "access_tags": list(self.access_tags),
        }

    def is_fresh(self, *, as_of: date | None = None) -> bool:
        """Whether this chunk should be retrievable right now.

        Excludes permanently-retired chunks (``obsolete``) and any outside its
        ``[effective_date, expiry_date]`` validity window. An empty date string means
        "no bound" on that side, so a chunk with no freshness metadata authored — the
        overwhelming majority of the corpus — is always fresh, unchanged from before
        this field existed.
        """

        today = as_of or date.today()
        if self.obsolete:
            return False
        if self.effective_date and date.fromisoformat(self.effective_date) > today:
            return False
        if self.expiry_date and date.fromisoformat(self.expiry_date) < today:
            return False
        return True


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by similarity search, with its relevance score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    """A source referenced by the generated answer."""

    tag: str  # e.g. "S1"
    document_id: str
    title: str
    doc_type: str
    source_path: str
    score: float


@dataclass(frozen=True)
class Answer:
    """The assistant's response to a question / log."""

    text: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    redactions: list[str] = field(default_factory=list)
    trace_id: str = ""
    # True only for the zero-chunk case: retrieval ran and found nothing, so the LLM
    # was never called. Other "no real answer" paths (empty query, guard block, empty KB)
    # are distinct, pre-existing conditions and stay False.
    abstained: bool = False
    # Deployed LLM/embedding/system-prompt versions that produced this answer, from
    # Settings.llm_model_version and RAGSettings.embedding_model_version/
    # prod_prompt_version. None only on the early-return paths in RAGEngine._answer()
    # that exit before the LLM (or, for embedding_version, retrieval) ever ran -- mirrors
    # the audit records' own "one exit point" scope boundary.
    llm_version: str | None = None
    embedding_version: str | None = None
    system_prompt_version: str | None = None
    # The sanitized text that actually drove retrieval, and the exact prompt sent
    # to the LLM -- both local-only inside RAGEngine._answer() before this, discarded
    # after use. Populated only at the same one exit point as the three version fields
    # above (None on every early return, and on a QA-cache hit for user_prompt
    # specifically -- see rag/engine.py). Never surfaced on the embedding application's
    # own response schema (its own field-by-field mapping from Answer); used only by an
    # application-level replay snapshot for root-cause debugging.
    retrieval_query: str | None = None
    user_prompt: str | None = None
    # A Turkish warning string when the retrieved chunk set itself contains
    # conflicting information on the same topic (e.g. an old vs. a new regulation) --
    # distinct from Faithfulness, which only checks whether the *answer* is
    # faithful to *some* retrieved chunk, not whether the chunks agree with each other.
    # None whenever detection is disabled, found nothing, or degraded silently on
    # failure (see RAGEngine._answer()).
    contradiction_warning: str | None = None

    @property
    def score(self) -> float | None:
        """Top retrieval confidence score, or ``None`` when nothing was retrieved.

        Derived from :attr:`retrieved` rather than stored separately -- ``retrieved``
        is already cleared to ``[]`` on every abstain/early-return path in
        ``RAGEngine._answer()``, so this stays consistent with ``abstained`` for free.
        """

        if not self.retrieved:
            return None
        return max(rc.score for rc in self.retrieved)
