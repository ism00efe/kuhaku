"""Rapid, time-bound benchmark sprint across multiple retrieval + LLM configurations.

Runs the 90-question ``eval/dataset.jsonl`` set through the full RAG pipeline
(retrieval -> optional rerank -> LLM generation via Ollama) for each configuration in the
table below, and reports ``recall@5``/``MRR`` for each. Built for a same-day presentation,
so the overriding constraint is that ONE slow or hung question must never stall the whole
run: every question gets a hard 40s budget, and a configuration that racks up too many
timeouts (or an in-process CUDA OOM) is abandoned in favor of moving on to the next one.

Usage:
    python kuhaku/scripts/benchmark_sprint.py

Requires:
  - The corpus already ingested into the default (RAGSettings.embedding_model) Chroma
    collection: `python scripts/ingest.py`.
  - A running Ollama server with `qwen2.5:7b` and `llama3.1:8b` pulled.
  - The two shootout embedding models and the reranker already downloaded locally
    (sentence-transformers pulls from the HF cache; nothing is fetched from the network
    here beyond what's already cached).

Design notes (see the code for the "why", not just the "what"):
  - recall@5/MRR are pure retrieval metrics -- unaffected by which LLM generates the final
    answer (see evaluation/metrics/functions.py/_recall_at_k,
    eval/evaluate.py/evaluate_retrieval). Config 6 and config 7 therefore report IDENTICAL
    recall@5/MRR by construction: they use the same
    retriever, differing only in which LLM turns the retrieved chunks into prose. Still run
    end-to-end (not retrieval-only) per explicit instruction, so the 40s timeout is measuring
    something real (LLM generation latency), and the table communicates this clearly.
  - Retrieval depth is forced to 5 (RAGEngine top_k=5) regardless of the configured
    RAGSettings.top_k, since recall@5 requires at least 5 retrieved chunks per question.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kuhaku.core.config import Settings, configure_logging, get_settings  # noqa: E402
from kuhaku.core.llm import build_llm_provider  # noqa: E402
from kuhaku.tools.rag import (  # noqa: E402
    ChromaVectorStore,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    RAGEngine,
    RAGSettings,
    Retriever,
    SentenceTransformerEmbeddings,
    build_bm25_from_corpus,
    build_chunker,
    ingest,
)
from kuhaku.tools.rag.prompts import SYSTEM_PROMPT  # noqa: E402

def load_dataset(path: Path) -> list[dict]:
    """Load labeled evaluation questions from a JSONL file (unknown keys are ignored)."""

    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


QUERY_TIMEOUT_SECONDS = 40
MAX_TIMEOUTS_PER_CONFIG = 3
DATASET_PATH = _ROOT / "eval" / "dataset.jsonl"
OUTPUT_PATH = _ROOT / "benchmark_results.md"

EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-large-instruct",
    "Qwen/Qwen3-Embedding-0.6B",
]
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
# "qwen2.5:7b" (no suffix) is not a tag Ollama actually publishes/has pulled here --
# "qwen2.5:7b-instruct" is what's locally available (and this repo's own default
# ollama_model in config.py), so that's what's used as the primary LLM.
PRIMARY_LLM = "qwen2.5:7b-instruct"
ALT_LLM = "llama3.1:8b"

LABEL_BM25 = "1. Sadece Kelime Araması (BM25)"
LABEL_DENSE = "2. Sadece Anlam Araması (Dense)"
LABEL_HYBRID = "3. Kelime + Anlam (Hybrid)"
LABEL_BM25_RERANK = "4. Kelime + Yeniden Sıralama"
LABEL_DENSE_RERANK = "5. Anlam + Yeniden Sıralama"
LABEL_ALL_DEFAULT = "6. Üçü Birlikte (Varsayılan)"
LABEL_ALL_LLAMA = "7. Üçü Birlikte (llama3.1:8b)"

TABLE_ROW_ORDER = [
    LABEL_BM25,
    LABEL_DENSE,
    LABEL_HYBRID,
    LABEL_BM25_RERANK,
    LABEL_DENSE_RERANK,
    LABEL_ALL_DEFAULT,
    LABEL_ALL_LLAMA,
]


@dataclass
class ConfigResult:
    name: str
    recall_at_5: float | None = None
    mrr: float | None = None
    notes: str = ""


def _slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_name).lower()


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    return "outofmemoryerror" in name or "out of memory" in text or "cuda oom" in text


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _score(doc_ids: list[str], expected: set[str]) -> tuple[float, float]:
    """(recall@5 hit, reciprocal rank) for one question, same definitions as
    eval/evaluate.py's evaluate_retrieval so numbers are comparable across scripts."""

    if not expected:
        return 0.0, 0.0
    hit = 1.0 if any(d in expected for d in doc_ids[:5]) else 0.0
    first_rank = next((i for i, d in enumerate(doc_ids, start=1) if d in expected), None)
    rr = 1.0 / first_rank if first_rank else 0.0
    return hit, rr


def get_or_build_embedder_and_store(
    rag_settings: RAGSettings, model_name: str, chunker
) -> tuple[SentenceTransformerEmbeddings, ChromaVectorStore]:
    """Embedder + its dedicated Chroma collection, ingesting the corpus into it first if
    that collection doesn't exist yet (mirrors service.py's embedding-switch collection
    naming: ``{collection_base}_{slug(model_name)}``, D46)."""

    embedder = SentenceTransformerEmbeddings(model_name, device=rag_settings.embedding_device)
    collection = f"{rag_settings.collection_base}_{_slug(model_name)}"
    store = ChromaVectorStore(rag_settings.chroma_persist_dir, collection)
    if store.count() == 0:
        print(f"  [setup] Ingesting corpus for '{model_name}' -> collection '{collection}'...")
        ingest(
            rag_settings.corpus_dir,
            embedder,
            store,
            chunk_size=rag_settings.chunk_size,
            overlap=rag_settings.chunk_overlap,
            reset=True,
            chunker=chunker,
        )
    return embedder, store


def build_reranker_or_none(rag_settings: RAGSettings) -> tuple[CrossEncoderReranker | None, bool]:
    """Loads the reranker onto its device up front (not lazily inside a query), so an OOM
    surfaces here -- once -- rather than being silently swallowed later by
    HybridRetriever's own per-query "degrade to un-reranked" error handling. Returns
    (reranker, oom_happened)."""

    reranker = CrossEncoderReranker(
        RERANKER_MODEL,
        retry_enabled=rag_settings.retry_enabled,
        retry_max_attempts=rag_settings.retry_reranker_max_attempts,
        retry_backoff_base_seconds=rag_settings.retry_reranker_backoff_base_seconds,
        retry_backoff_max_seconds=rag_settings.retry_reranker_backoff_max_seconds,
    )
    try:
        reranker._ensure_model()  # noqa: SLF001 -- deliberate warm-up, see docstring
        return reranker, False
    except Exception as exc:
        if _is_oom(exc):
            print(f"  [setup] Reranker OOM while loading '{RERANKER_MODEL}': {exc}")
            _empty_cuda_cache()
            return None, True
        raise


def run_configuration(
    label: str,
    settings: Settings,
    default_embedder,
    default_store,
    dataset: list[dict],
    retriever: Retriever,
    llm_model: str,
    extra_note: str = "",
) -> ConfigResult:
    """Replays every question in ``dataset`` through a full RAGEngine.answer() call built
    from ``retriever``/``llm_model``, each capped at QUERY_TIMEOUT_SECONDS. Aborts early
    (recall/MRR left as None) on >3 timeouts or an in-process CUDA OOM.
    """

    print(f"\n=== {label} (LLM={llm_model}) ===")
    result = ConfigResult(name=label, notes=extra_note)
    started = time.monotonic()

    run_settings = settings.model_copy(
        update={"llm_provider": "ollama", "ollama_model": llm_model}
    )
    llm = build_llm_provider(run_settings)
    engine = RAGEngine(
        default_embedder,
        default_store,
        llm,
        top_k=5,  # recall@5 needs >=5 retrieved chunks, independent of RAGSettings.top_k
        retriever=retriever,
        audit_log_path=settings.audit_log_path,
        system_prompt=SYSTEM_PROMPT,
    )

    recall_hits = 0.0
    reciprocal_ranks: list[float] = []
    timeouts = 0
    completed = 0
    n = len(dataset)

    try:
        for i, item in enumerate(dataset, start=1):
            question = item["question"]
            expected = set(item["expected_sources"])
            preview = question[:60]

            # A fresh single-worker executor per question: a hung call must never block
            # later questions from being submitted, which a shared pool would do.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(engine.answer, question)
                answer = future.result(timeout=QUERY_TIMEOUT_SECONDS)
                doc_ids = [c.chunk.document_id for c in answer.retrieved]
                hit, rr = _score(doc_ids, expected)
                recall_hits += hit
                reciprocal_ranks.append(rr)
                completed += 1
                print(f"  [{i:>2}/{n}] ok      rr={rr:.2f}  {preview}")
            except concurrent.futures.TimeoutError:
                timeouts += 1
                reciprocal_ranks.append(0.0)
                print(f"  [{i:>2}/{n}] TIMEOUT (>{QUERY_TIMEOUT_SECONDS}s)  {preview}")
                if timeouts >= MAX_TIMEOUTS_PER_CONFIG:
                    result.notes = (result.notes + " >3 TIMEOUTS").strip()
                    print(f"  -> aborting '{label}': {timeouts} timeouts")
                    return result
            except Exception as exc:
                if _is_oom(exc):
                    raise
                reciprocal_ranks.append(0.0)
                print(f"  [{i:>2}/{n}] ERROR: {exc}")
            finally:
                # wait=False: the background thread (if still running) is left to finish
                # or fail on its own -- it is bounded by the LLM provider's own request
                # timeout, never by us. We must not block waiting for it here.
                executor.shutdown(wait=False)
    except Exception as exc:
        if _is_oom(exc):
            _empty_cuda_cache()
            result.notes = (result.notes + " OOM").strip()
            print(f"  -> aborting '{label}': OOM ({exc})")
            return result
        raise

    result.recall_at_5 = recall_hits / n if n else None
    result.mrr = sum(reciprocal_ranks) / n if n else None
    elapsed_min = (time.monotonic() - started) / 60
    print(
        f"  -> recall@5={result.recall_at_5:.3f}  MRR={result.mrr:.3f}  "
        f"({completed}/{n} completed, {timeouts} timeouts, {elapsed_min:.1f} min)"
    )
    return result


def render_markdown_table(results: dict[str, ConfigResult]) -> str:
    lines = [
        "| Configuration | recall@5 | MRR | Notes |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for label in TABLE_ROW_ORDER:
        r = results.get(label)
        if r is None:
            lines.append(f"| **{label}** | - | - | not run |")
            continue
        recall_cell = f"{r.recall_at_5:.3f}" if r.recall_at_5 is not None else (r.notes or "N/A")
        mrr_cell = f"{r.mrr:.3f}" if r.mrr is not None else (r.notes or "N/A")
        lines.append(f"| **{label}** | {recall_cell} | {mrr_cell} | {r.notes} |")
    return "\n".join(lines)


def main() -> None:
    configure_logging(level="WARNING")
    settings = get_settings()
    rag_settings = RAGSettings.from_settings(settings)
    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} evaluation questions from {DATASET_PATH}")

    default_embedder = SentenceTransformerEmbeddings(
        rag_settings.embedding_model, device=rag_settings.embedding_device
    )
    default_store = ChromaVectorStore(
        rag_settings.chroma_persist_dir, rag_settings.chroma_collection
    )
    if default_store.count() == 0:
        raise SystemExit(
            "Vector store is empty. Run `python scripts/ingest.py` first "
            f"(collection '{rag_settings.chroma_collection}')."
        )

    chunker = build_chunker(rag_settings)
    embedder_cache: dict[str, tuple] = {
        rag_settings.embedding_model: (default_embedder, default_store)
    }

    def embedder_for(model_name: str):
        if model_name not in embedder_cache:
            embedder_cache[model_name] = get_or_build_embedder_and_store(
                rag_settings, model_name, chunker
            )
        return embedder_cache[model_name]

    results: dict[str, ConfigResult] = {}
    overall_start = time.monotonic()

    try:
        # --- Step 1: embedding shootout (config #2, run twice) -----------------------
        shootout: dict[str, ConfigResult] = {}
        for model_name in EMBEDDING_MODELS:
            try:
                embedder, store = embedder_for(model_name)
                retriever = DenseRetriever(embedder, store)
                shootout[model_name] = run_configuration(
                    f"{LABEL_DENSE} [{model_name}]",
                    settings, embedder, store, dataset, retriever, PRIMARY_LLM,
                )
            except Exception as exc:
                print(f"Embedding shootout failed for '{model_name}': {exc}")
                shootout[model_name] = ConfigResult(name=model_name, notes=f"failed: {exc}")

        scored = {m: r for m, r in shootout.items() if r.recall_at_5 is not None}
        best_model = max(scored, key=lambda m: scored[m].recall_at_5) if scored else EMBEDDING_MODELS[0]
        comparison = ", ".join(
            f"{m.split('/')[-1]}={r.recall_at_5:.3f}" if r.recall_at_5 is not None
            else f"{m.split('/')[-1]}={r.notes or 'FAIL'}"
            for m, r in shootout.items()
        )
        best_result = shootout[best_model]
        best_result.notes = f"best={best_model.split('/')[-1]} ({comparison})"
        results[LABEL_DENSE] = best_result
        print(f"\n>>> Best embedding: {best_model}\n")

        best_embedder, best_store = embedder_cache[best_model]
        embedding_note = f"embedding={best_model}"

        # --- Step 2: fill the rest of the table ---------------------------------------
        sparse = build_bm25_from_corpus(
            rag_settings.corpus_dir,
            chunk_size=rag_settings.chunk_size,
            overlap=rag_settings.chunk_overlap,
            k1=rag_settings.bm25_k1,
            b=rag_settings.bm25_b,
            chunker=chunker,
            rag_settings=rag_settings,
        )

        results[LABEL_BM25] = run_configuration(
            LABEL_BM25, settings, default_embedder, default_store, dataset, sparse, PRIMARY_LLM,
        )

        dense = DenseRetriever(best_embedder, best_store)
        hybrid = HybridRetriever(
            dense, sparse, None,
            rrf_k=rag_settings.rrf_k, candidates=rag_settings.rerank_candidates,
            max_chunks_per_document=rag_settings.max_chunks_per_document,
        )
        results[LABEL_HYBRID] = run_configuration(
            LABEL_HYBRID, settings, best_embedder, best_store, dataset, hybrid, PRIMARY_LLM,
            extra_note=embedding_note,
        )

        # Reranker is loaded (warmed up) once and, if it fits, shared across configs
        # 4/5/6/7 -- an OOM here disables it for all four remaining rows in one shot
        # rather than re-attempting (and re-OOMing) four separate model loads.
        reranker, reranker_oom = build_reranker_or_none(rag_settings)

        if reranker_oom:
            for label in (LABEL_BM25_RERANK, LABEL_DENSE_RERANK, LABEL_ALL_DEFAULT, LABEL_ALL_LLAMA):
                results[label] = ConfigResult(name=label, notes="OOM (reranker)")
        else:
            # Passing `sparse` in the `dense` slot is deliberate: HybridRetriever only
            # ever calls .retrieve() on whatever occupies that slot, so with sparse=None
            # this yields exactly "BM25 -> rerank", with no dense/embedding step at all.
            bm25_rerank = HybridRetriever(
                sparse, None, reranker,
                rrf_k=rag_settings.rrf_k, candidates=rag_settings.rerank_candidates,
                max_chunks_per_document=rag_settings.max_chunks_per_document,
            )
            results[LABEL_BM25_RERANK] = run_configuration(
                LABEL_BM25_RERANK, settings, default_embedder, default_store,
                dataset, bm25_rerank, PRIMARY_LLM,
            )

            dense_rerank = HybridRetriever(
                dense, None, reranker,
                rrf_k=rag_settings.rrf_k, candidates=rag_settings.rerank_candidates,
                max_chunks_per_document=rag_settings.max_chunks_per_document,
            )
            results[LABEL_DENSE_RERANK] = run_configuration(
                LABEL_DENSE_RERANK, settings, best_embedder, best_store,
                dataset, dense_rerank, PRIMARY_LLM, extra_note=embedding_note,
            )

            all_three = HybridRetriever(
                dense, sparse, reranker,
                rrf_k=rag_settings.rrf_k, candidates=rag_settings.rerank_candidates,
                max_chunks_per_document=rag_settings.max_chunks_per_document,
            )
            results[LABEL_ALL_DEFAULT] = run_configuration(
                LABEL_ALL_DEFAULT, settings, best_embedder, best_store,
                dataset, all_three, PRIMARY_LLM, extra_note=embedding_note,
            )
            # Same retriever instance as config 6 -- retrieval is identical, only the LLM
            # changes, so recall@5/MRR are expected to match config 6 exactly (see the
            # module docstring). Not a bug: these two metrics can't see the LLM at all.
            results[LABEL_ALL_LLAMA] = run_configuration(
                LABEL_ALL_LLAMA, settings, best_embedder, best_store,
                dataset, all_three, ALT_LLM,
                extra_note=f"{embedding_note}; recall/MRR identical to config 6 by "
                "construction (same retriever, LLM has no effect on retrieval)",
            )
    finally:
        elapsed_min = (time.monotonic() - overall_start) / 60
        table = render_markdown_table(results)
        print("\n" + table)
        OUTPUT_PATH.write_text(table + "\n", encoding="utf-8")
        print(f"\nResults written to {OUTPUT_PATH}")
        print(f"Script finished in {elapsed_min:.1f} minutes")


if __name__ == "__main__":
    main()
