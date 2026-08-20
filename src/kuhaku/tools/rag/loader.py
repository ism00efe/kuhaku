"""Document loaders for RAG ingestion pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from kuhaku.tools.rag.models import Document
from kuhaku.core.sanitization import sanitize

from .config import RAGSettings
from .ingestion import (
    _TEXT_SUFFIXES,
    _freshness_fields,
    _infer_doc_type,
    _infer_title,
    _unique_id,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class DocumentLoader(Protocol):
    """Protocol for document loaders."""

    def load(self, source: str) -> list[Document]:
        """Load documents from a source (e.g., path, URL)."""
        ...


class FileSystemLoader:
    """Document loader that reads supported text files from a directory on disk."""

    def __init__(
        self,
        directory_path: str | None = None,
        *,
        rag_settings: RAGSettings | None = None,
        doc_type_mapping: dict[str, str] | None = None,
    ) -> None:
        """``doc_type_mapping``, when given, wins over ``rag_settings.doc_type_prefix_mapping``
        -- an explicit mapping is more specific than a whole settings object. Neither
        given (the default) means every loaded document's ``doc_type`` falls back to
        ``"document"`` (see :func:`kuhaku.tools.rag.ingestion._infer_doc_type`)."""

        self.directory_path = directory_path
        if doc_type_mapping is not None:
            self._doc_type_mapping = doc_type_mapping
        elif rag_settings is not None:
            self._doc_type_mapping = rag_settings.doc_type_prefix_mapping
        else:
            self._doc_type_mapping = None

    def load(self, source: str) -> list[Document]:
        dir_path = source or self.directory_path
        if not dir_path:
            raise ValueError("Directory path must be provided either in __init__ or load().")

        root = Path(dir_path)
        if not root.exists():
            raise FileNotFoundError(
                f"Corpus directory '{dir_path}' not found. "
                "Run `python scripts/generate_data.py` first."
            )

        documents: list[Document] = []
        seen_ids: set[str] = set()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            clean = sanitize(raw)
            documents.append(
                Document(
                    id=_unique_id(path, seen_ids),
                    title=_infer_title(clean, path.stem),
                    doc_type=_infer_doc_type(path.name, self._doc_type_mapping),
                    text=clean,
                    source_path=str(path.relative_to(root)),
                    **_freshness_fields(clean),
                )
            )
        logger.info("Loaded %d documents from %s", len(documents), dir_path)
        return documents
