"""Corpus ingestion: load -> sanitize -> chunk -> embed -> index.

Two entry points populate the vector store. ``ingest`` is the offline pipeline that reads
every file under a directory (the corpus loaded at startup); it is intentionally agnostic
about whether the corpus is synthetic or real, and swapping in real enterprise documents
means pointing ``CORPUS_DIR`` elsewhere. ``ingest_single_document`` is the online
counterpart the upload API calls: same sanitize -> chunk -> embed steps, but for one
already-in-memory document appended to an existing collection, never touching disk.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import filetype

from kuhaku.tools.rag.models import Chunk, Document
from kuhaku.core.sanitization import Redaction, sanitize_text
from kuhaku.core.security import inspect_query

from .chunking import Chunker, ParagraphChunker, chunk_document  # noqa: F401 - re-exported
from .config import RAGSettings
from .embeddings import EmbeddingProvider
from .messages import DEFAULT_ENGINE_MESSAGES, EngineMessages
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class UnsupportedFileType(ValueError):
    """Uploaded content doesn't match an accepted, content-verified file type."""


class EmptyContent(ValueError):
    """Uploaded content — or the text extracted from it — is empty."""


class UploadTooLarge(ValueError):
    """Uploaded content exceeds the configured maximum size."""

_TEXT_SUFFIXES = {".md", ".txt", ".json", ".xml", ".log"}
_UPLOAD_SUFFIXES = _TEXT_SUFFIXES | {".pdf"}


def _extract_pdf_text(content: bytes, messages: EngineMessages) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        # pypdf raises several distinct error types (PdfReadError, and plain ValueError /
        # KeyError from malformed structures) for a corrupt or non-standard PDF. All of
        # them mean the same thing to this caller: the bytes could not be read as a PDF.
        raise UnsupportedFileType(
            messages.pdf_read_failed_template.format(error=exc)
        ) from exc
    return "\n\n".join(p for p in pages if p.strip())


def extract_text(
    filename: str,
    content: bytes,
    *,
    max_upload_bytes: int | None = None,
    messages: EngineMessages | None = None,
) -> str:
    """Validate an uploaded file's content and extract its plain text.

    Validated by content, not the filename's extension. ``filetype.guess`` inspects magic
    bytes: a claimed ``.pdf`` must carry a real PDF signature, and content matching some
    *other* concrete binary signature (image, archive, executable, ...) is rejected
    outright, since none of the accepted formats are meant to be binary. Plain-text
    formats have no signature of their own — ``filetype.guess`` returns ``None`` for them
    by design — so those are instead validated by requiring a clean UTF-8 decode, which
    rejects binary garbage renamed with a text extension the same way.

    ``max_upload_bytes``, when given, is enforced against the raw content length before
    any other validation or parsing runs (a corrupt/oversized upload should never reach
    ``filetype.guess`` or ``pypdf``). ``None`` (the default) means no limit is enforced.
    ``messages``, when given, supplies every user-facing string this function can raise;
    ``None`` (the default) falls back to :data:`DEFAULT_ENGINE_MESSAGES`.
    """

    messages = messages if messages is not None else DEFAULT_ENGINE_MESSAGES

    if max_upload_bytes is not None and len(content) > max_upload_bytes:
        raise UploadTooLarge(
            messages.upload_size_exceeded_template.format(max_bytes=max_upload_bytes)
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        allowed = ", ".join(sorted(_UPLOAD_SUFFIXES))
        raise UnsupportedFileType(
            messages.unsupported_file_type_template.format(
                suffix=suffix or "(no extension)", allowed=allowed
            )
        )
    if not content:
        raise EmptyContent(messages.empty_file)

    kind = filetype.guess(content)
    if kind is not None:
        if suffix != ".pdf" or kind.extension != "pdf":
            raise UnsupportedFileType(
                messages.content_extension_mismatch_template.format(
                    detected=kind.extension, suffix=suffix
                )
            )
        text = _extract_pdf_text(content, messages)
    elif suffix == ".pdf":
        raise UnsupportedFileType(messages.missing_pdf_signature)
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedFileType(messages.undecodable_content) from exc

    if not text.strip():
        raise EmptyContent(messages.no_extractable_text)
    return text


# FR4: the document's optional "**Metadata:** key: value · key: value · ..." line
# (see D20/D30 for why this stays prose, not YAML frontmatter). Matches the whole line
# so any key/value set can be parsed once and reused by whichever fields a caller wants.
_METADATA_LINE_RE = re.compile(r"^\*\*Metadata:\*\*\s*(.+)$", re.MULTILINE)


def _parse_metadata_block(text: str) -> dict[str, str]:
    """Extract ``key: value`` pairs from the document's "**Metadata:**" line, if any."""

    match = _METADATA_LINE_RE.search(text)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for segment in match.group(1).split("·"):
        key, sep, value = segment.strip().partition(":")
        if sep:
            fields[key.strip().lower()] = value.strip()
    return fields


class _FreshnessFields(TypedDict):
    effective_date: str
    obsolete: bool
    expiry_date: str


def _freshness_fields(text: str) -> _FreshnessFields:
    """Pull FR4's optional freshness fields out of the Metadata line.

    Absent for essentially every document today (no existing corpus doc authors these
    keys) -- the empty-string / False defaults below mean such a document is always
    fresh, exactly as before this field existed. A ``TypedDict`` (not a plain
    ``dict[str, str | bool]``) so ``Document(..., **_freshness_fields(text))`` type-checks
    each keyword against ``Document``'s actual field types.
    """

    fields = _parse_metadata_block(text)
    return _FreshnessFields(
        effective_date=fields.get("effective_date", ""),
        obsolete=fields.get("obsolete", "").strip().lower() == "true",
        expiry_date=fields.get("expiry_date", ""),
    )


def _normalize_access_tags(access_tags: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and normalize caller-supplied access tags at the ingestion boundary.

    `None` and an empty sequence both collapse to the same "untagged" representation
    (`()`), so tagging stays a deliberate, opt-in act. A blank (empty or whitespace-only)
    tag is rejected outright rather than stored as something that can never match a
    caller's roles. Duplicate tags are harmless -- silently deduplicated (order
    preserved), since a repeated tag changes nothing about who can see the chunk.
    """

    if not access_tags:
        return ()
    seen: dict[str, None] = {}
    for tag in access_tags:
        if not tag or not tag.strip():
            raise ValueError("access_tags: tag must not be empty or whitespace-only")
        seen.setdefault(tag, None)
    return tuple(seen)


def _infer_doc_type(filename: str, mapping: dict[str, str] | None = None) -> str:
    """Infer a document's type from ``mapping`` (filename prefix -> doc_type), falling
    back to "document" when ``mapping`` is ``None``/empty or no prefix matches. Callers
    that have a :class:`~kuhaku.tools.rag.config.RAGSettings` pass
    ``rag_settings.doc_type_prefix_mapping`` explicitly -- this function has no
    ambient/global settings fallback of its own."""

    prefix_mapping = mapping or {}
    for prefix, doc_type in prefix_mapping.items():
        if filename.startswith(prefix):
            return doc_type
    return "document"


def _infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:80]
    return fallback


def _unique_id(path: Path, seen: set[str]) -> str:
    """Derive a stable, collision-free document id from a file path.

    The id is the filename stem, which keeps ids readable and stable. Two files can share
    a stem (``report.pdf`` / ``report.docx``, or our ``log_err_x.json`` / ``.xml``), so a
    collision falls back to appending the extension, then a counter. Without this, both
    documents would produce identical chunk ids and silently overwrite each other in the
    vector store. Iteration order is sorted, so ids are deterministic across runs.
    """

    doc_id = path.stem
    if doc_id in seen:
        doc_id = f"{path.stem}_{path.suffix.lstrip('.').lower()}"
        counter = 2
        base = doc_id
        while doc_id in seen:  # pragma: no cover - needs a stem *and* stem+ext collision
            doc_id = f"{base}_{counter}"
            counter += 1
    seen.add(doc_id)
    return doc_id


from .loader import DocumentLoader, FileSystemLoader  # noqa: E402


def load_corpus(corpus_dir: str, rag_settings: RAGSettings | None = None) -> list[Document]:
    """Read every supported file under ``corpus_dir`` into a :class:`Document`.

    Documents are sanitized here, once, before chunking — guaranteeing no sensitive
    value can reach the chunker, embedder, or vector store. ``rag_settings``, when
    given, supplies ``doc_type_prefix_mapping`` for doc-type inference (see
    :class:`FileSystemLoader`); ``None`` (the default) means every loaded document's
    ``doc_type`` falls back to ``"document"``.
    """

    loader = FileSystemLoader(rag_settings=rag_settings)
    return loader.load(corpus_dir)


def ingest(
    corpus_dir: str,
    embedder: EmbeddingProvider,
    store: VectorStore,
    *,
    chunk_size: int,
    overlap: int,
    reset: bool = True,
    chunker: Chunker | None = None,
    loader: DocumentLoader | None = None,
    access_tags: Sequence[str] | None = None,
) -> int:
    """Run the full ingestion pipeline. Returns the number of chunks indexed.

    ``access_tags``, when given, is applied uniformly to every document this call loads
    from ``corpus_dir`` -- per-file tagging is not supported here, same as
    :meth:`~kuhaku.RAG.load_documents`. ``None`` (the default) leaves every document
    untagged, unchanged from before this parameter existed.
    """

    if reset:
        store.reset()

    chunker = chunker or ParagraphChunker()
    loader = loader or FileSystemLoader()
    documents = loader.load(corpus_dir)

    normalized_tags = _normalize_access_tags(access_tags)
    if normalized_tags:
        documents = [
            dataclasses.replace(doc, access_tags=normalized_tags) for doc in documents
        ]

    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc, chunk_size=chunk_size, overlap=overlap))

    if not chunks:
        logger.warning("No chunks produced from corpus '%s'.", corpus_dir)
        return 0

    logger.info("Embedding %d chunks...", len(chunks))
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.add(chunks, embeddings)
    logger.info("Indexed %d chunks from %d documents.", len(chunks), len(documents))
    return len(chunks)


def ingest_single_document(
    text: str,
    filename: str,
    embedder: EmbeddingProvider,
    store: VectorStore,
    *,
    chunk_size: int,
    overlap: int,
    chunker: Chunker | None = None,
    rag_settings: RAGSettings | None = None,
    max_upload_bytes: int | None = None,
    messages: EngineMessages | None = None,
    access_tags: Sequence[str] | None = None,
) -> tuple[int, list[Redaction]]:
    """Sanitize, chunk, embed, and add ONE document to an existing collection.

    The online counterpart to :func:`ingest`, used by the upload API. Never touches disk
    and never resets the store — ``store.add`` only appends, so the corpus loaded at
    startup is untouched. Both ``text`` and ``filename`` are sanitized: the filename ends
    up in stored metadata (title fallback, ``source_path``) and is operator-supplied, not
    developer-curated like a corpus path, so it gets the same guarantee as document text.

    The id gets a random suffix rather than reusing the filename stem the way corpus
    ingestion does: corpus ids only need to be unique within one directory listing, but an
    upload lands in a collection that may already hold any id, and colliding would either
    overwrite an existing document's chunks or fail outright depending on the store.

    ``rag_settings``, when given, supplies ``doc_type_prefix_mapping`` for
    :func:`_infer_doc_type`; ``None`` (the default) means ``doc_type`` falls back to
    ``"document"`` for every upload.

    ``max_upload_bytes``, when given, is enforced against ``text``'s UTF-8 encoded byte
    length before sanitization or chunking run -- this is the only size guard available
    to callers (e.g. :meth:`~kuhaku.tools.rag.engine.RAGEngine.ingest_document`) that
    hand this function already-extracted text rather than the raw upload bytes
    :func:`extract_text` checks. ``None`` (the default) means no limit is enforced.
    ``messages``, when given, supplies every user-facing string this function can raise;
    ``None`` (the default) falls back to :data:`DEFAULT_ENGINE_MESSAGES`.

    ``access_tags``, when given, gates every chunk this document produces (document-level
    access filtering; see ``kuhaku.tools.rag.retriever.is_entitled``). ``None`` (the
    default) leaves the document untagged -- visible to any caller, unchanged from before
    this parameter existed. A blank tag is rejected outright (see
    ``_normalize_access_tags``) so a typo can never silently produce an unprotected
    document.
    """

    messages = messages if messages is not None else DEFAULT_ENGINE_MESSAGES
    normalized_tags = _normalize_access_tags(access_tags)

    if max_upload_bytes is not None and len(text.encode("utf-8")) > max_upload_bytes:
        raise UploadTooLarge(
            messages.upload_size_exceeded_template.format(max_bytes=max_upload_bytes)
        )

    clean_filename, filename_redactions = sanitize_text(filename)
    clean_text, text_redactions = sanitize_text(text)
    redactions = filename_redactions + text_redactions

    safe, reason = inspect_query(clean_text)
    if not safe:
        raise ValueError(messages.document_security_check_failed_template.format(reason=reason))

    doc_type_mapping = rag_settings.doc_type_prefix_mapping if rag_settings is not None else None
    doc = Document(
        id=f"upload_{uuid4().hex[:12]}",
        title=_infer_title(clean_text, Path(clean_filename).stem),
        doc_type=_infer_doc_type(clean_filename, doc_type_mapping),
        text=clean_text,
        source_path=f"uploads/{clean_filename}",
        access_tags=normalized_tags,
        **_freshness_fields(clean_text),
    )
    chunker = chunker or ParagraphChunker()
    chunks = chunker.chunk(doc, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0, redactions

    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.add(chunks, embeddings)
    logger.info("Indexed %d chunks from uploaded document '%s'.", len(chunks), doc.id)
    return len(chunks), redactions
