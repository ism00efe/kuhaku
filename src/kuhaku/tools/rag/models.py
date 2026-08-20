"""RAG-specific domain models.

Plain dataclasses with no external dependencies. These are the vocabulary the RAG
pipeline speaks (documents, chunks, retrieval results, citations, answers). They live
under ``tools/rag`` (not ``kuhaku/core``) because they are specific to the RAG tool --
a different tool built on kuhaku has no reason to know what a ``Chunk`` is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Document:
    """A source document in the knowledge base (synthetic or, later, real)."""

    id: str
    title: str
    doc_type: str
    text: str
    source_path: str
    metadata: dict[str, str] = field(default_factory=dict)
    # FR4 freshness metadata, parsed from the document's prose Metadata line by
    # rag/ingestion.py. "" means "no bound" on that side -- the overwhelming majority
    # of documents author none of this and are always fresh, unchanged from before.
    effective_date: str = ""
    obsolete: bool = False
    expiry_date: str = ""


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
    # FR4 freshness metadata, inherited from the parent Document at chunk time. Kept as
    # plain str/bool (never None) because ChromaDB metadata values must be str/int/
    # float/bool -- see DECISIONS.md D35.
    effective_date: str = ""
    obsolete: bool = False
    expiry_date: str = ""

    def metadata(self) -> dict[str, str | int | bool]:
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
        }

    def is_fresh(self, *, as_of: date | None = None) -> bool:
        """Whether this chunk should be retrievable right now (FR4).

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
    # True only for the FR1 zero-chunk case: retrieval ran and found nothing, so the LLM
    # was never called. Other "no real answer" paths (empty query, guard block, empty KB)
    # are distinct, pre-existing conditions and stay False -- see DECISIONS.md D32.
    abstained: bool = False
    # D42: deployed LLM/embedding/system-prompt versions that produced this answer, from
    # Settings.llm_model_version and RAGSettings.embedding_model_version/
    # prod_prompt_version. None only on the early-return paths in RAGEngine._answer()
    # that exit before the LLM (or, for embedding_version, retrieval) ever ran -- see
    # D42 for the exact scope boundary, mirroring D41's "one exit point" precedent for
    # audit records.
    llm_version: str | None = None
    embedding_version: str | None = None
    system_prompt_version: str | None = None
    # D49: the sanitized text that actually drove retrieval, and the exact prompt sent
    # to the LLM -- both local-only inside RAGEngine._answer() before this, discarded
    # after use. Populated only at the same one exit point as the three D42 fields above
    # (None on every early return, and on a QA-cache hit for user_prompt specifically --
    # see rag/engine.py). Never surfaced on AnalyzeResponse (api/schemas.py's
    # from_answer() is an explicit field-by-field mapping); used only by the D49 replay
    # snapshot (evaluation/replay_storage.py) for root-cause debugging.
    retrieval_query: str | None = None
    user_prompt: str | None = None
    # D50: a Turkish warning string when the retrieved chunk set itself contains
    # conflicting information on the same topic (e.g. an old vs. a new regulation) --
    # distinct from Faithfulness (D47), which only checks whether the *answer* is
    # faithful to *some* retrieved chunk, not whether the chunks agree with each other.
    # None whenever detection is disabled, found nothing, or degraded silently on
    # failure (see RAGEngine._answer() and DECISIONS.md D50).
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
