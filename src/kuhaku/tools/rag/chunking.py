"""Chunking strategies: split a sanitized Document into indexable Chunks.

Two chunkers satisfy the same small ``Chunker`` protocol, selected via
``RAGSettings.chunking_strategy`` (see ``build_chunker``):

- ``ParagraphChunker`` (default): fixed-size, paragraph-packed character windows,
  with no structural awareness. This is the original, unchanged behavior.
- ``StructuralChunker``: splits along Markdown heading hierarchy (#, ##, ###) and
  pulls Markdown tables and "Term: Definition" glossary lines into their own
  chunks (``content_type`` "table" / "glossary") instead of packing them as prose.
  See DECISIONS.md D28.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from kuhaku.tools.rag.config import RAGSettings
from kuhaku.tools.rag.models import Chunk, Document


class Chunker(Protocol):
    """Splits a sanitized Document into indexable Chunks."""

    def chunk(self, doc: Document, *, chunk_size: int, overlap: int) -> list[Chunk]: ...


def _pack_paragraphs(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into ``chunk_size``-char windows, hard-splitting any
    single paragraph too long to fit, with an ``overlap``-char stride."""

    windows: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= chunk_size:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            if buffer:
                windows.append(buffer)
            if len(para) <= chunk_size:
                buffer = para
            else:
                start = 0
                while start < len(para):
                    windows.append(para[start : start + chunk_size])
                    start += max(1, chunk_size - overlap)
                buffer = ""
    if buffer:
        windows.append(buffer)
    return windows


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split a document into overlapping, paragraph-aware character windows."""

    text = doc.text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    windows = _pack_paragraphs(paragraphs, chunk_size, overlap)

    return [
        Chunk(
            id=f"{doc.id}::{i}",
            document_id=doc.id,
            title=doc.title,
            doc_type=doc.doc_type,
            text=window,
            chunk_index=i,
            source_path=doc.source_path,
            effective_date=doc.effective_date,
            obsolete=doc.obsolete,
            expiry_date=doc.expiry_date,
        )
        for i, window in enumerate(windows)
    ]


class ParagraphChunker:
    """Default chunker: paragraph-aware fixed windows (today's behavior, unchanged)."""

    def chunk(self, doc: Document, *, chunk_size: int, overlap: int) -> list[Chunk]:
        return chunk_document(doc, chunk_size, overlap)


# --- StructuralChunker -------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")

# Term must not itself contain a colon (rules out "14:30: message" jumping to the
# second colon) and the definition must be separated by a space (rules out "https://",
# "14:30" with no following space). Length/word caps and the stopword list filter
# annotation-style prefixes ("Not:", "Örnek:") that aren't glossary terms.
_GLOSSARY_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s+)?\*{0,2}(?P<term>[^\n:*]{1,40}?)\*{0,2}\s*:\s+(?P<definition>\S.*)$"
)
_GLOSSARY_STOPWORDS = {
    "not", "note", "örnek", "example", "önemli", "important",
    "uyarı", "warning", "dikkat", "tip", "see", "also", "ayrıca", "nb",
}


def _split_by_headings(text: str) -> list[tuple[list[str], str]]:
    """Return (heading breadcrumb, section body) tuples, in document order.

    Only levels 1-3 (#, ##, ###) are split boundaries; a deeper heading (####+) is
    left as ordinary body text. A document with no headings at all yields exactly
    one section with an empty breadcrumb and the whole text as body -- "preamble
    before the first heading" and "no headings anywhere" are the same code path.
    """

    lines = text.splitlines()
    stack: list[tuple[int, str]] = []
    current_body: list[str] = []
    sections: list[tuple[list[str], list[str]]] = []

    def flush() -> None:
        if any(line.strip() for line in current_body):
            sections.append(([title for _, title in stack], list(current_body)))

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_body.clear()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        else:
            current_body.append(line)
    flush()

    return [(breadcrumb, "\n".join(body).strip()) for breadcrumb, body in sections]


def _is_markdown_table(paragraph: str) -> bool:
    """A real Markdown table has a header row immediately followed by a dash
    separator row -- requiring both, not just a stray ``|``, is what keeps prose
    containing a literal pipe (e.g. a shell command sample) from being misread."""

    lines = [line for line in paragraph.splitlines() if line.strip()]
    return (
        len(lines) >= 2
        and bool(_TABLE_ROW_RE.match(lines[0]))
        and bool(_TABLE_SEP_RE.match(lines[1]))
    )


def _parse_markdown_table(paragraph: str) -> dict:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        cells = (cells + [""] * len(header))[: len(header)]  # defensive: ragged rows
        rows.append(dict(zip(header, cells, strict=False)))
    return {"columns": header, "rows": rows}


def _match_glossary(line: str) -> dict[str, str] | None:
    match = _GLOSSARY_LINE_RE.match(line)
    if not match:
        return None
    term = match.group("term").strip()
    definition = match.group("definition").strip()
    if not term or not term[0].isalpha():
        return None
    if len(term.split()) > 5:
        return None
    if term.lower() in _GLOSSARY_STOPWORDS:
        return None
    return {"term": term, "definition": definition}


def _batch_glossary(
    entries: list[dict[str, str]], chunk_size: int
) -> list[list[dict[str, str]]]:
    """One glossary chunk per section by default; split further only if the
    serialized JSON would exceed ``chunk_size`` -- same bound as prose windowing."""

    if not entries:
        return []
    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_len = 2  # "[]"
    for entry in entries:
        entry_len = len(json.dumps(entry, ensure_ascii=False)) + 1
        if current and current_len + entry_len > chunk_size:
            batches.append(current)
            current, current_len = [], 2
        current.append(entry)
        current_len += entry_len
    if current:
        batches.append(current)
    return batches


class StructuralChunker:
    """Splits a document by Markdown heading hierarchy, extracting tables and
    glossary entries into their own ``content_type``-tagged chunks instead of
    packing everything as undifferentiated prose."""

    def chunk(self, doc: Document, *, chunk_size: int, overlap: int) -> list[Chunk]:
        text = doc.text.strip()
        if not text:
            return []

        pieces: list[tuple[str, str]] = []  # (content_type, chunk_text), in order

        for breadcrumb, body in _split_by_headings(text):
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            prose_paragraphs: list[str] = []
            glossary_entries: list[dict[str, str]] = []

            for para in paragraphs:
                if _is_markdown_table(para):
                    table = _parse_markdown_table(para)
                    pieces.append(("table", json.dumps(table, ensure_ascii=False)))
                    continue

                prose_lines: list[str] = []
                for line in para.splitlines():
                    entry = _match_glossary(line)
                    if entry is not None:
                        glossary_entries.append(entry)
                    else:
                        prose_lines.append(line)
                remaining = "\n".join(prose_lines).strip()
                if remaining:
                    prose_paragraphs.append(remaining)

            heading_prefix = " > ".join(breadcrumb)
            for window in _pack_paragraphs(prose_paragraphs, chunk_size, overlap):
                chunk_text = f"{heading_prefix}\n\n{window}" if heading_prefix else window
                pieces.append(("text", chunk_text))

            for batch in _batch_glossary(glossary_entries, chunk_size):
                pieces.append(("glossary", json.dumps(batch, ensure_ascii=False)))

        return [
            Chunk(
                id=f"{doc.id}::{i}",
                document_id=doc.id,
                title=doc.title,
                doc_type=doc.doc_type,
                text=chunk_text,
                chunk_index=i,
                source_path=doc.source_path,
                content_type=content_type,
                effective_date=doc.effective_date,
                obsolete=doc.obsolete,
                expiry_date=doc.expiry_date,
            )
            for i, (content_type, chunk_text) in enumerate(pieces)
        ]


def build_chunker(settings: RAGSettings) -> Chunker:
    """Instantiate the chunker selected by ``settings.chunking_strategy``."""

    strategy = settings.chunking_strategy.strip().lower()
    if strategy == "structural":
        return StructuralChunker()
    if strategy == "paragraph":
        return ParagraphChunker()
    raise ValueError(
        f"Unknown CHUNKING_STRATEGY '{settings.chunking_strategy}'. "
        "Expected one of: paragraph, structural."
    )
