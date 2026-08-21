# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

### Added

- Initial standalone release of **kuhaku**, a provider-agnostic AI orchestration
  framework: a generic runtime core plus RAG as its first reference tool.
- **`kuhaku.core`** — generic, tool-agnostic infrastructure:
  - Typed, environment-driven configuration (`Settings`, `get_settings`, nested
    `KUHAKU_RAG__*` overrides).
  - An `LLMProvider` protocol with Ollama, Anthropic, OpenAI, and Google Vertex AI
    implementations, selected via `LLM_PROVIDER` through `build_llm_provider`.
  - Authentication/authorization primitives (`AuthContext`, `AuthorizationPolicy`,
    API-key and JWT `AuthProvider`s), opt-in and unconfigured by default.
  - Deterministic, regex-based PII sanitization (`sanitize_text`) covering emails,
    tokens, IPs, credit card numbers, Turkish national IDs (TCKN), and phone numbers.
  - A prompt-injection guard pipeline (`GuardPipeline`, `inspect_query`) and an
    unconditional audit log.
  - Structured JSON logging with trace-id propagation, and OpenTelemetry tracing +
    metrics (`instrumented_step`), exported to Prometheus.
  - Retry-with-backoff and circuit breakers (`call_with_retry`) shared by every
    external call site (LLM, embeddings, vector store, re-ranker).
  - A three-tier startup fallback/strictness policy for optional components
    (`core.policy`).
- **`kuhaku.tools.rag`** — the reference RAG pipeline:
  - Document ingestion (`.txt`, `.md`, `.pdf`) with paragraph and structural chunking.
  - Sentence-transformer and Vertex AI embedding providers, backed by a ChromaDB
    vector store.
  - Dense, BM25 sparse, and RRF-fused hybrid retrieval, with optional multilingual
    cross-encoder re-ranking.
  - LLM-based query rewriting with caching, and contradiction detection across
    retrieved chunks.
  - A query-answer cache, and `RAGEngine` orchestrating the full
    sanitize → guard → retrieve → generate → cite → audit pipeline behind the
    `EmbeddingProvider` / `VectorStore` / `LLMProvider` / `Retriever` protocols.
- **`kuhaku.evaluation`** — a tool-agnostic evaluation harness:
  - A golden-dataset loader (`load_eval_dataset`) and `EvaluationTarget` /
    `TargetAdapter` contract usable against `RAGEngine` or any other target.
  - Retrieval metrics (hit rate@k, MRR, nDCG@k, precision@k, recall@k) and
    answer-quality metrics (faithfulness via an LLM judge, answer correctness).
  - `EvaluationRunner`, with in-memory and SQLite result stores.
- **`kuhaku.RAG`** — a top-level facade over `RAGEngine` construction, exposing
  `ingest()`, `load_documents()`, `ask()`, and an interactive `chat_repl()`, for
  callers who don't need full composition-root control.
- Full test suite (`tests/`, 900+ tests) covering the RAG pipeline, security,
  observability, auth, and evaluation layers against in-memory fakes.
