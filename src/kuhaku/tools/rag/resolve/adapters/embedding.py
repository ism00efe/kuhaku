"""Embedding-backend candidates (§7). Resolved independently of the LLM.

Two kinds:
  - local: the ``sentence-transformers`` package installed *and* the model present on
    disk. No network liveness check. What is embedded is the full text of every document
    on every ingest -- but it never leaves the machine, so ``sends_document_text`` is
    False here.
  - API-based: opted into explicitly via ``embedding_provider`` plus that provider's own
    credential. An LLM API key never routes here.
"""

from __future__ import annotations

from collections.abc import Sequence

from kuhaku.core.resolve import Candidate, Cost, Environment
from kuhaku.core.resolve.probes import hf_cache_roots, module_available

_LOCAL_ID = "sentence-transformer"
_VERTEX_ID = "vertex-embeddings"
# First-run download for the shipped default model family; an estimate, not a measurement.
_LOCAL_DOWNLOAD_BYTES = 490_000_000


def local_candidate_id() -> str:
    return _LOCAL_ID


def _model_cached(model_name: str) -> bool:
    """True when the model is already downloaded in any of the HuggingFace /
    sentence-transformers cache locations, with an actual snapshot present -- not just a
    ``models--*`` directory left behind by an interrupted download."""

    slug = model_name.replace("/", "--")
    for root in hf_cache_roots():
        snapshots = root / f"models--{slug}" / "snapshots"
        try:
            if snapshots.is_dir() and any(snapshots.iterdir()):
                return True
        except OSError:
            continue
    return False


class LocalEmbeddingAdapter:
    kind = "embedding"
    packages = frozenset({"sentence-transformers", "torch"})

    def __init__(self, rag_settings, *, build) -> None:
        self._rs = rag_settings
        self._build = build

    def probe(self, env: Environment) -> Sequence[Candidate]:
        rs = self._rs
        have_pkg = module_available("sentence_transformers")
        cached = _model_cached(rs.embedding_model)
        needs: list[str] = []
        if not have_pkg:
            needs.append("pip install 'kuhaku[dense]'")
        if not cached:
            needs.append("a first-run model download (~490 MB)")
        note = "; ".join(needs) if needs else "package installed, model on disk"
        return [Candidate(
            id=_LOCAL_ID,
            kind="embedding",
            label=f"local embeddings ({rs.embedding_model})",
            cost=Cost(
                install_required=not have_pkg,
                download_required=not cached,
                download_bytes=_LOCAL_DOWNLOAD_BYTES,
                sends_document_text=False,
                note=note,
            ),
            ready=have_pkg and cached,
            safety_rank=0,
            activate=self._build,
        )]

    def baseline(self, env: Environment) -> None:
        return None


class ApiEmbeddingAdapter:
    kind = "embedding"
    packages = frozenset()

    def __init__(self, rag_settings, *, build) -> None:
        self._rs = rag_settings
        self._build = build

    def probe(self, env: Environment) -> Sequence[Candidate]:
        rs = self._rs
        # Only when the operator has explicitly chosen an API embedder AND supplied its
        # own configuration -- an LLM key contributes nothing here (§7).
        if rs.embedding_provider.strip().lower() != "vertex" or not rs.vertex_project:
            return []
        return [Candidate(
            id=_VERTEX_ID,
            kind="embedding",
            label="Vertex AI embeddings",
            cost=Cost(
                network_per_use=True,
                monetary=True,
                sends_document_text=True,
                note="every ingested document's full text is sent to Google Cloud",
            ),
            ready=True,
            safety_rank=8,
            activate=self._build,
        )]

    def baseline(self, env: Environment) -> None:
        return None
