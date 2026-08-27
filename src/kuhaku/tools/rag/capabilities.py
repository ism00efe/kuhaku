"""RAG-specific resolution of ``"auto"`` settings.

The RAG counterpart to :mod:`kuhaku.core.capabilities`, mirroring how
:class:`~kuhaku.tools.rag.config.RAGSettings` is the RAG counterpart to
:class:`~kuhaku.core.config.Settings`: the generic probes and the notice mechanism live
in core, and the chains that are only meaningful for retrieval live here.

Two ``"auto"`` settings are RAG-owned:

  - ``embedding_device`` -- ``cuda``/``mps``/``cpu``, from what torch reports.
  - ``retrieval`` -- ``hybrid`` when a real embedding provider can be built, ``sparse``
    (BM25 only, no embeddings, no torch, no model download) when it cannot. That second
    decision needs the actual build attempt, so it is made in ``RAG.__init__``; this
    module only owns the announcement (:func:`announce_retrieval_downgrade`).
"""

from __future__ import annotations

from kuhaku.core.capabilities import (
    AUTO,
    Resolution,
    auto_enabled,
    emit,
    torch_accelerator,
)


def resolve_embedding_device(configured: str | None) -> str:
    """``"cpu"`` | ``"cuda"`` | ``"mps"``.

    A concrete value the caller pinned (via ``KUHAKU_RAG__EMBEDDING_DEVICE`` or
    ``RAGSettings(embedding_device=...)``) is absolute. ``"auto"`` (the default) asks
    torch what is available; ``KUHAKU_AUTO=false`` forces ``"cpu"``.
    """

    if configured not in (None, AUTO):
        return configured
    if not auto_enabled():
        return "cpu"
    device = torch_accelerator()
    emit(Resolution("embedding_device", device, "cpu", f"torch reports '{device}'"))
    return device


def announce_retrieval_downgrade(reason: str) -> None:
    """Tell the operator, on the terminal, that ``retrieval="auto"`` resolved to
    ``"sparse"`` because a dense/embedding backend could not be built."""

    emit(Resolution("retrieval", "sparse", "hybrid", reason))
