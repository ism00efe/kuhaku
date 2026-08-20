"""Quick retrieval-only benchmark: recall@5 and MRR for six retrieval strategies.

recall@5/MRR are pure retrieval metrics -- they depend only on which chunks a retriever
returns, never on the LLM that eventually reads them (see
evaluation/metrics/functions.py:_recall_at_k/_mrr and eval/evaluate.py:evaluate_retrieval,
both of which score off retrieved chunk ids alone). So this script never calls the LLM:
each of the 90 eval/dataset.jsonl questions runs straight through a retriever.retrieve()
call.

The 30s budget is PER QUESTION, not per configuration: each retriever.retrieve() call gets
its own fresh single-worker executor and its own 30s ceiling, so one slow question can
never consume another question's budget. Worst case is therefore 90 x 30s = 45 minutes for
a single configuration; a question that times out scores 0 for both recall@5 and MRR (it
never had a retrieved-chunks answer to score) and the sweep moves on to the next question.

Uses the embedding model already configured/ingested (multilingual-e5-small by default --
no re-ingestion, no embedding shootout) and jinaai/jina-reranker-v2-base-multilingual.

Usage:
    python kuhaku/scripts/benchmark_quick.py
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kuhaku.core.config import configure_logging, get_settings  # noqa: E402
from kuhaku.tools.rag.models import RetrievedChunk  # noqa: E402
from kuhaku.tools.rag import (  # noqa: E402
    ChromaVectorStore,
    DenseRetriever,
    HybridRetriever,
    RAGSettings,
    Retriever,
    SentenceTransformerEmbeddings,
    build_bm25_from_corpus,
    build_chunker,
)

QUESTION_TIMEOUT_SECONDS = 30
PROGRESS_EVERY = 15
DATASET_PATH = _ROOT / "eval" / "dataset.jsonl"
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"


def load_dataset(path: Path) -> list[dict]:
    """Load labeled evaluation questions from a JSONL file (unknown keys are ignored)."""

    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class _FastEmbedReranker:
    """fastembed's ONNX-backed TextCrossEncoder, forced onto CPU.

    Swapped in for jina-reranker-v2-base-multilingual specifically because
    sentence-transformers' own CrossEncoder can only load this particular model via
    trust_remote_code=True (arbitrary code execution from the HF repo, which is not
    something to enable even scoped to one benchmark script). fastembed ships this
    model as a pre-exported ONNX graph instead, so no remote Python code ever runs --
    same reranking model, no code-execution trust decision required.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self._model_name, providers=["CPUExecutionProvider"])
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates or top_k <= 0:
            return []
        model = self._ensure_model()
        documents = [item.chunk.text for item in candidates]
        scores = list(model.rerank(query, documents))
        rescored = [
            RetrievedChunk(chunk=item.chunk, score=float(score))
            for item, score in zip(candidates, scores, strict=True)
        ]
        # Ties break on chunk id, matching CrossEncoderReranker's own convention, so
        # ordering stays deterministic.
        rescored.sort(key=lambda r: (-r.score, r.chunk.id))
        return rescored[:top_k]


CONFIG_ORDER = [
    "sparse_only",
    "dense_only",
    "hybrid_no_rerank",
    "sparse_with_rerank",
    "dense_with_rerank",
    "hybrid_with_rerank",
]


def _score(doc_ids: list[str], expected: set[str]) -> tuple[float, float]:
    if not expected:
        return 0.0, 0.0
    hit = 1.0 if any(d in expected for d in doc_ids[:5]) else 0.0
    first_rank = next((i for i, d in enumerate(doc_ids, start=1) if d in expected), None)
    rr = 1.0 / first_rank if first_rank else 0.0
    return hit, rr


def _evaluate(retriever: Retriever, dataset: list[dict]) -> tuple[float, float]:
    """Score every question, each capped at QUESTION_TIMEOUT_SECONDS independently.

    A fresh single-worker executor per question -- same reasoning as
    tests/benchmark_sprint.py: a hung/slow call must never block the next question from
    being submitted, which a shared pool would do.
    """

    n = len(dataset)
    recall_hits = 0.0
    reciprocal_ranks: list[float] = []
    for i, item in enumerate(dataset, start=1):
        expected = set(item["expected_sources"])
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(retriever.retrieve, item["question"], 5)
            retrieved = future.result(timeout=QUESTION_TIMEOUT_SECONDS)
            doc_ids = [r.chunk.document_id for r in retrieved]
            hit, rr = _score(doc_ids, expected)
        except concurrent.futures.TimeoutError:
            hit, rr = 0.0, 0.0
        finally:
            # wait=False: a still-running call is left to finish on its own; we must
            # not block waiting for it once its 30s budget is spent.
            executor.shutdown(wait=False)
        recall_hits += hit
        reciprocal_ranks.append(rr)
        if i % PROGRESS_EVERY == 0 or i == n:
            print(f"  {i}/{n} done")
    return recall_hits / n, sum(reciprocal_ranks) / n


def main() -> None:
    configure_logging(level="WARNING")
    _empty_cuda_cache()
    settings = get_settings()
    rag_settings = RAGSettings.from_settings(settings)
    dataset = load_dataset(DATASET_PATH)

    embedder = SentenceTransformerEmbeddings(
        rag_settings.embedding_model, device=rag_settings.embedding_device
    )
    store = ChromaVectorStore(rag_settings.chroma_persist_dir, rag_settings.chroma_collection)
    if store.count() == 0:
        raise SystemExit("Vector store is empty. Run `python scripts/ingest.py` first.")
    chunker = build_chunker(rag_settings)

    dense = DenseRetriever(embedder, store)
    sparse = build_bm25_from_corpus(
        rag_settings.corpus_dir,
        chunk_size=rag_settings.chunk_size,
        overlap=rag_settings.chunk_overlap,
        k1=rag_settings.bm25_k1,
        b=rag_settings.bm25_b,
        chunker=chunker,
        rag_settings=rag_settings,
    )

    def rrf(dense_r: Retriever | None, sparse_r: Retriever | None, reranker) -> Retriever:
        # Whichever of dense_r/sparse_r is non-None occupies the required `dense` slot --
        # HybridRetriever only ever calls .retrieve() on it, so passing BM25 there with
        # sparse_r=None yields plain "BM25 (+ optional rerank)", no embedding step at all.
        primary = dense_r if dense_r is not None else sparse_r
        secondary = sparse_r if dense_r is not None else None
        return HybridRetriever(
            primary, secondary, reranker,
            rrf_k=rag_settings.rrf_k,
            candidates=rag_settings.rerank_candidates,
            max_chunks_per_document=rag_settings.max_chunks_per_document,
        )

    retrievers: dict[str, Retriever | None] = {
        "sparse_only": sparse,
        "dense_only": dense,
        "hybrid_no_rerank": rrf(dense, sparse, None),
    }

    reranker_error: str | None = None
    try:
        reranker = _FastEmbedReranker(RERANKER_MODEL)
        reranker._ensure_model()  # noqa: SLF001 -- warm up now, not mid-sweep
        retrievers["sparse_with_rerank"] = rrf(sparse, None, reranker)
        retrievers["dense_with_rerank"] = rrf(dense, None, reranker)
        retrievers["hybrid_with_rerank"] = rrf(dense, sparse, reranker)
    except Exception as exc:
        reranker_error = str(exc)
        for name in ("sparse_with_rerank", "dense_with_rerank", "hybrid_with_rerank"):
            retrievers[name] = None

    results: dict[str, tuple[str, str]] = {}
    for name in CONFIG_ORDER:
        retriever = retrievers.get(name)
        if retriever is None:
            results[name] = ("OOM", "OOM")
            continue
        print(f"=== {name} ===")
        recall, mrr = _evaluate(retriever, dataset)
        results[name] = (f"{recall:.3f}", f"{mrr:.3f}")

    if reranker_error:
        print(f"# reranker load failed, *_with_rerank configs marked OOM: {reranker_error}")
    print("recall@5 for:")
    for name in CONFIG_ORDER:
        print(f"{name}={results[name][0]}")
    print()
    print("MRR for:")
    for name in CONFIG_ORDER:
        print(f"{name}={results[name][1]}")


if __name__ == "__main__":
    main()
