# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

First public release. Alpha: the API may change before 1.0.

### Added

**`kuhaku.core`** — tool-agnostic runtime infrastructure.

- Typed, environment-driven configuration (`Settings`, `get_settings`), read from
  `KUHAKU_`-prefixed variables with RAG settings nested under `KUHAKU_RAG__`. Four
  ecosystem-standard names (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) are accepted unprefixed.
- An `LLMProvider` protocol with Ollama, Anthropic, OpenAI and Google Vertex AI
  implementations, selected through `build_llm_provider`.
- Identity and authorization primitives (`AuthContext`, `AuthorizationPolicy`, API-key and
  JWT providers), independent of any tool.
- Deterministic PII sanitization covering emails, tokens, IP addresses, card numbers,
  national ID numbers and phone numbers.
- A deterministic prompt-injection input guard, and an audit log that records one entry
  per request whatever the outcome.
- Structured logging with trace-id propagation, plus OpenTelemetry tracing and metrics
  exported through Prometheus.
- Retry with exponential backoff and circuit breakers, shared by every external call site.

**`kuhaku.tools.rag`** — the first tool built on core.

- Document ingestion (`.txt`, `.md`, `.pdf`) with paragraph and structural chunking.
- Sentence-transformer and Vertex AI embedding providers over a persistent ChromaDB store.
- Dense, BM25 sparse and RRF-fused hybrid retrieval — hybrid by default — with optional
  multilingual cross-encoder re-ranking, off by default.
- **Document-level access filtering.** Chunks carry `access_tags`; a caller's
  `AuthContext.roles` must intersect them. Untagged chunks are visible to everyone;
  tagged chunks require a matching identity. Enforced before ranking in every strategy, so
  an entitled caller still receives a full result set.
- A query-answer cache keyed by the entitled chunk set, so two entitlements never share an
  entry.
- A layered system prompt: a framework-owned safety core (instruction precedence, data
  marking, canary, grounding, mandatory citations, contradiction handling) plus
  caller-owned persona, output-language policy, format preference and worked example, all
  with neutral English defaults.
- `RAG`, a facade over `RAGEngine` construction exposing `ingest()`, `load_documents()`,
  `ask()` and `chat_repl()`, with `rag.engine` as the escape hatch for full
  constructor-injection control.

**`kuhaku.evaluation`** — a tool-agnostic evaluation harness.

- A golden-dataset loader and an `EvaluationTarget` contract usable against `RAGEngine` or
  any other target.
- Retrieval metrics (hit rate@k, MRR, nDCG@k, precision@k, recall@k) and answer-quality
  metrics (faithfulness via an LLM judge, answer correctness).
- `EvaluationRunner` with in-memory and SQLite result stores.

### Not included

Query rewriting, contradiction detection and the layered prompt-injection guard v2 ship in
the package but are not wired to the facade, and guard v2 needs a model you train
yourself — no weights are distributed. They are reachable by constructing `RAGEngine`
directly and are not recommended for this release.

`load_documents` applies one tag set to the whole call rather than per file. The BM25 index
rebuilds on the first query after ingestion rather than updating incrementally. There are
no async APIs, and Chroma is the only vector store implementation.

### Notes

1023 tests, all passing and entirely offline — in-memory fakes for the embedding provider,
vector store and LLM. No network access, model download or running LLM server is required
to run the suite.
