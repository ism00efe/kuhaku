# kuhaku

kuhaku is a provider-agnostic AI orchestration framework. It provides a generic,
tool-agnostic runtime core — LLM abstraction, security, observability, authentication,
retry/resilience, and an evaluation harness — plus **RAG (retrieval-augmented
generation)** as its first reference tool, built entirely on top of that core.

The core makes no assumption about what tool is running on it; the RAG tool makes no
assumption about where its documents come from. Either layer can be used on its own.

## Package layout

```
kuhaku/
├── core/            generic, tool-agnostic infrastructure
│   ├── config.py        typed, env-driven Settings
│   ├── llm/              LLMProvider interface + Ollama/Anthropic/OpenAI/Vertex AI backends
│   ├── auth/              AuthContext, AuthorizationPolicy, API-key/JWT providers
│   ├── security/          prompt-injection guard pipeline, output checks, audit log
│   ├── observability/     structured logging, OpenTelemetry tracing + metrics
│   ├── sanitization.py    deterministic, regex-based PII masking
│   ├── retry.py           retry-with-backoff + circuit breaker
│   ├── policy.py          startup fallback/strictness policy for optional components
│   └── models.py          generic domain primitives (Message, ToolCall, ExecutionResult)
├── evaluation/      tool-agnostic evaluation harness (dataset, metrics, runner, stores)
└── tools/
    └── rag/         the RAG tool: ingestion, chunking, embeddings, vector store,
                      retrieval, re-ranking, query rewriting, contradiction detection,
                      caching, and RAGEngine
```

A tool is only ever allowed to depend on `kuhaku.core`, never the other way around —
adding a second tool alongside RAG means giving it its own `tools/<name>` package, not
adding tool-specific fields to `core`.

## Installation

Requires Python 3.11+.

```bash
# from the repository root, editable install
pip install -e ./kuhaku

# with the optional Google Vertex AI provider (LLM + embeddings)
pip install -e "./kuhaku[vertex]"

# with development tooling (pytest, ruff, mypy)
pip install -e "./kuhaku[dev]"
```

The default LLM provider is a local [Ollama](https://ollama.com) server
(`qwen2.5:7b-instruct`) — nothing external is required to try kuhaku out. Switch
providers with the `LLM_PROVIDER` environment variable (see [Configuration](#configuration)).

## Quickstart

```python
from kuhaku import RAG

rag = RAG()  # dense retrieval, local Ollama, no re-ranking — all overridable
rag.load_documents("path/to/docs")  # indexes every .txt / .md / .pdf in the directory

answer = rag.ask("What does error code 96 mean?")
print(answer.text)
for citation in answer.citations:
    print(f"  [{citation.tag}] {citation.title} (score={citation.score:.3f})")
```

`RAG` also exposes `ingest(text, filename)` for indexing a single in-memory document, and
`chat_repl()` for an interactive terminal chat loop.

Common knobs, all optional keyword arguments to `RAG(...)`:

```python
rag = RAG(
    retrieval="hybrid",             # "dense" | "sparse" | "hybrid"
    reranker=True,                  # False | True | a HuggingFace cross-encoder model name
    corpus_dir="path/to/docs",      # required for "sparse"/"hybrid" (BM25 is rebuilt from disk)
    vector_store="path/to/chroma",  # persistent Chroma dir; defaults to a temp directory
    embedding="intfloat/multilingual-e5-small",
)
```

For anything the facade doesn't expose (query rewriting, contradiction detection,
caching, the security guard, auth), either construct `RAGEngine` directly (see
[Beyond the facade](#beyond-the-facade)) or reach through the escape hatch:

```python
rag.engine.update_authorization_policy(my_policy)
answer = rag.engine.answer(question, auth_context=my_auth_context)
```

## Configuration

Settings are typed (`pydantic-settings`) and read from process environment variables,
plus an optional `.env` file a caller opts into explicitly — kuhaku never assumes a
current working directory. All fields have defaults; nothing is required to run the
local Ollama default.

```python
from kuhaku import get_settings, Settings

settings = get_settings()                 # cached, environment-driven
settings = Settings(_env_file=".env")     # or load a specific dotenv file
```

Key environment variables:

```bash
LLM_PROVIDER=ollama            # ollama | anthropic | openai | vertex
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...

# RAG-specific settings nest under the KUHAKU_RAG__ prefix (see RAGSettings)
KUHAKU_RAG__TOP_K=8
KUHAKU_RAG__RERANK_ENABLED=true
KUHAKU_RAG__CHUNK_SIZE=500
KUHAKU_RAG__CHUNKING_STRATEGY=paragraph   # paragraph | structural
```

`RAGSettings` (`kuhaku.tools.rag.RAGSettings`) is usable entirely standalone — it has
its own sensible defaults and does not require a `Settings` instance.

## Beyond the facade

`RAG` wraps `RAGEngine` behind a handful of named parameters. For full
composition-root control, construct the pieces directly — this is exactly what `RAG`
does internally:

```python
from kuhaku.core.config import Settings
from kuhaku.core.llm import build_llm_provider
from kuhaku.tools.rag import (
    ChromaVectorStore, DenseRetriever, RAGEngine, RAGSettings,
    build_chunker, build_embedding_provider,
)

settings = Settings()
rag_settings = RAGSettings.from_settings(settings)

embedder = build_embedding_provider(rag_settings)
store = ChromaVectorStore(rag_settings.chroma_persist_dir, rag_settings.chroma_collection)
llm = build_llm_provider(settings)

engine = RAGEngine(
    embedder, store, llm,
    top_k=rag_settings.top_k,
    retriever=DenseRetriever(embedder, store),
    chunker=build_chunker(rag_settings),
    rag_settings=rag_settings,
)

answer = engine.answer("What does error code 96 mean?")
```

`RAGEngine` depends only on the `EmbeddingProvider`, `VectorStore`, `LLMProvider`, and
`Retriever` protocols — never a concrete SDK — so any of them can be swapped or faked
independently (see `tests/conftest.py` for in-memory fakes used by the test
suite).

### Retrieval

- **Dense** — embedding similarity search against the Chroma store (`DenseRetriever`).
- **Sparse** — BM25 keyword search, rebuilt from `corpus_dir` on construction
  (`SparseRetriever` / `build_bm25_from_corpus`).
- **Hybrid** — dense + sparse fused with Reciprocal Rank Fusion (`HybridRetriever`).
- **Re-ranking** — an optional cross-encoder pass (`CrossEncoderReranker`) over the top
  candidates from any of the above; use a multilingual model if your queries and corpus
  are in different languages.

### Beyond retrieval

- **Query rewriting** (`QueryRewriter`) — optional, LLM-based, cached query rewriting
  ahead of retrieval.
- **Contradiction detection** (`ContradictionDetector`) — flags when the retrieved
  chunks disagree with each other on the same topic (distinct from faithfulness, which
  checks the *answer* against *some* retrieved chunk).
- **Query-answer cache** (`QueryAnswerCache`) — SQLite-backed cache keyed on the
  sanitized query + retrieval configuration.

## Security

- **Sanitization** (`kuhaku.core.sanitization.sanitize_text`) — deterministic,
  regex-based PII masking, run before embedding and before any LLM call, on both the
  ingestion and query paths. Masks emails, tokens, IPs, credit card numbers (Luhn-
  validated) and Turkish national IDs / TCKN (checksum-validated), and phone numbers.
  The LLM is never asked to sanitize.
- **Prompt-injection guard** (`kuhaku.core.security.GuardPipeline` / `inspect_query`) —
  a deterministic, regex/classifier-based guard pipeline (normalize → two-stage
  classify → `"pass"` / `"restricted"` / `"reject"` zone decision), independent of any
  LLM call.
- **Audit log** (`kuhaku.core.security.record_audit` / `read_audit_log`) — every
  request is recorded regardless of guard configuration.
- **Auth** (`kuhaku.core.auth`) — `AuthContext`, `AuthorizationPolicy`, and
  `AuthProvider` (API key / JWT) primitives. kuhaku defines no role or permission
  vocabulary itself; nothing here is wired in automatically — an unconfigured policy
  means "allow everything," unchanged from before the package existed.

## Observability

- **Structured logging** — JSON log lines with a `trace_id` propagated through
  `contextvars`, no signature changes required at call sites.
- **OpenTelemetry tracing + metrics** — `kuhaku.core.observability.instrumented_step`
  opens a span, times a pipeline stage, logs it, and records the duration on a
  histogram, in a single `with` block:

  ```python
  from kuhaku.core.observability import instrumented_step

  with instrumented_step("retrieve") as rec:
      retrieved = engine.retrieve(query)
      rec.set(chunk_count=len(retrieved))
  ```

- Metrics are exported to Prometheus via `opentelemetry-exporter-prometheus`.

## Evaluation

`kuhaku.evaluation` is a tool-agnostic evaluation harness: it speaks only in terms of
`EvaluationTarget`/`TargetAdapter` (anything exposing `evaluate_sample()`, or
`ask()`/`answer()` and `retrieve()`/`search()`), so it works against `RAGEngine` or any
other target.

```python
from kuhaku.evaluation import EvaluationRunner, HitRateAtKMetric, MRRMetric

runner = EvaluationRunner(
    metrics=[HitRateAtKMetric(k=5), MRRMetric()],
    dataset_path="path/to/eval_dataset.jsonl",  # JSONL: question_id, question, expected_sources, ...
)
summary = runner.run(rag.engine)  # dict[str, float] of aggregate scores
```

Built-in metrics: retrieval (`hit_rate_at_k`, `mrr`, `ndcg_at_k`, `precision_at_k`,
`recall_at_k`) and answer quality (`faithfulness`, `answer_correctness`, with an
LLM-judge-based faithfulness evaluator). Results can be persisted via
`InMemoryEvaluationStore` or `SqliteEvaluationStore`.

## Resilience

Every external call site (LLM providers, the embedding provider, the vector store, the
re-ranker) goes through the same retry-with-backoff and circuit-breaker logic
(`kuhaku.core.retry.call_with_retry`), configured independently per subsystem. On
exhaustion, the original exception is re-raised unchanged — retry is a policy layered
on top of each component's own error handling, not a replacement for it.

## Testing

```bash
pytest
```

The suite runs entirely against in-memory fakes for the embedding provider, vector
store, and LLM (see `tests/conftest.py`) — no external network, model download,
or running LLM server is required.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
