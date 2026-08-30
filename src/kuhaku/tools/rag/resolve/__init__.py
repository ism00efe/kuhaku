"""RAG-owned capability resolution: the adapters that contribute ``embedding``,
``store`` and ``retrieval`` candidates, plus the policy that acts on a chosen store.

The generic loop lives in :mod:`kuhaku.core.resolve`; this package only adds RAG's
candidate lists and RAG-scoped policy, never a branch in core.
"""

from __future__ import annotations
