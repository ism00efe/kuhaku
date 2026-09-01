"""Vector-store candidates. Only Chroma today: the built-in lightweight (Tier-0) store
and Qdrant are follow-up work (see ARCHITECTURE.md). Until the Tier-0 store lands, this
is the sole candidate, so the resolver always takes the "exactly one" branch.
"""

from __future__ import annotations

from collections.abc import Sequence

from kuhaku.core.resolve import Candidate, Cost, Environment
from kuhaku.core.resolve.probes import module_available


class StoreAdapter:
    kind = "store"
    packages = frozenset({"chromadb"})

    def __init__(self, rag_settings, *, build) -> None:
        self._rs = rag_settings
        self._build = build

    def probe(self, env: Environment) -> Sequence[Candidate]:
        have = module_available("chromadb")
        return [Candidate(
            id="chroma",
            kind="store",
            label="Chroma (serverless, on-disk)",
            cost=Cost(
                install_required=not have,
                note="pip install chromadb" if not have else "on disk, no network",
            ),
            ready=have,
            safety_rank=1,
            activate=self._build,
        )]

    def baseline(self, env: Environment) -> Candidate:
        return Candidate(
            id="chroma", kind="store", label="Chroma",
            cost=Cost(note="documented baseline"),
            ready=module_available("chromadb"), safety_rank=1,
            activate=self._build,
        )
