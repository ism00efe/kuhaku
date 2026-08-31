"""RAG-owned capability resolution: the adapters that contribute ``embedding`` and
``store`` candidates, plus the policy that acts on a chosen store
(:mod:`kuhaku.tools.rag.resolve.store_policy`).

The generic loop lives in :mod:`kuhaku.core.resolve`; this package only adds RAG's
candidate lists and RAG-scoped policy, never a branch in core.
"""

from __future__ import annotations

from kuhaku.core.resolve import Registry

from .adapters.embedding import (
    ApiEmbeddingAdapter,
    LocalEmbeddingAdapter,
    local_candidate_id,
    model_on_disk,
)
from .adapters.store import StoreAdapter

__all__ = ["build_rag_registry", "local_candidate_id", "model_on_disk"]


def build_rag_registry(rag_settings, *, build_embedder, build_store) -> Registry:
    """A registry with RAG's embedding and store adapters.

    ``build_embedder`` / ``build_store`` are zero-arg callables that construct the
    backend -- ``RAG.__init__`` passes closures over the module-level builder names so
    tests that patch those keep working, and so the "how to build it" detail stays out
    of the adapters.
    """

    registry = Registry()
    registry.register(LocalEmbeddingAdapter(rag_settings, build=build_embedder))
    registry.register(ApiEmbeddingAdapter(rag_settings, build=build_embedder))
    registry.register(StoreAdapter(rag_settings, build=build_store))
    return registry
