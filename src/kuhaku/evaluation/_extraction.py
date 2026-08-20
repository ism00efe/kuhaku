"""Normalize a target's retrieved items into plain ids/texts.

Shared by the retrieval metrics, the faithfulness metric, and the runner (for the
``retrieved_ids`` it hands to ``EvaluationStore``). Accepts either real
``RetrievedChunk`` objects or bare strings, so a lightweight fake retriever in a test can
return ``["doc-1", "doc-2"]`` without constructing full chunk objects.
"""

from __future__ import annotations

from typing import Any


def extract_document_ids(retrieved: list[Any]) -> list[str]:
    ids: list[str] = []
    for item in retrieved:
        if isinstance(item, str):
            ids.append(item)
        elif hasattr(item, "chunk") and hasattr(item.chunk, "document_id"):
            ids.append(item.chunk.document_id)
        else:
            raise TypeError(f"cannot extract a document id from retrieved item: {item!r}")
    return ids


def extract_texts(retrieved: list[Any]) -> list[str]:
    texts: list[str] = []
    for item in retrieved:
        if isinstance(item, str):
            texts.append(item)
        elif hasattr(item, "chunk") and hasattr(item.chunk, "text"):
            texts.append(item.chunk.text)
        else:
            texts.append(str(item))
    return texts
