# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Environment-aware `"auto"` settings. `retrieval`, `llm_provider` and
  `embedding_device` now ship as `"auto"` and are resolved once, at construction, from
  what is installed or reachable: `llm_provider` prefers a reachable Ollama then the
  first provider whose credentials are set; `embedding_device` picks CUDA/MPS/CPU from
  torch; `retrieval` stays `hybrid` when an embedding backend can be built and downgrades
  to `sparse` (BM25 only — no embeddings, no torch, no model download) when it cannot,
  announcing the downgrade on stderr and as a `FallbackWarning`. An explicit value is
  always absolute, `"auto"` never triggers a download, and `KUHAKU_AUTO=false` disables
  the probing entirely. New modules: `kuhaku.core.capabilities` (tool-agnostic probes +
  resolver) and `kuhaku.tools.rag.capabilities` (the RAG chains); `NullEmbeddings` backs
  the sparse-only path.
- `SECURITY.md`: vulnerability reporting policy, so GitHub's Security tab surfaces one.
- CI/CD/PyPI/license badges in `README.md`.
- CI's `docs` job now deploys the built site to GitHub Pages (`mkdocs gh-deploy`) on
  every push to `main`; every job (including PRs) still runs `mkdocs build --strict`
  first, so a broken internal link still fails the build without publishing anything.
- `RAG(...)` now accepts any field name from `Settings` or `RAGSettings` that has no
  dedicated named parameter (e.g. `RAG(guard_enabled=True)`, `RAG(chunk_size=999)`,
  `RAG(llm_timeout_seconds=30)`) and applies it as an override, the same as building a
  `Settings`/`RAGSettings` instance yourself and passing it as `settings=`/
  `rag_settings=`. A name matching neither still raises `TypeError`, so a typo is
  caught exactly as before. This means a new field added to either settings class is
  usable through `RAG()` immediately, with no change to `RAG.__init__` itself.

### Changed

- `RAG()` now validates two things at construction instead of discovering them lazily
  at the first `ask()`/`ingest()`: `guard_enabled=True` with no `GuardPipeline` wired in
  (`RAG()` does not build one itself) now raises `SecurityComponentError` immediately,
  since the facade cannot silently honor a setting the caller explicitly turned on; an
  unwritable audit log path now logs a warning at construction (with the reason)
  instead of only a per-request warning once the first record fails to write.
  `kuhaku.core.policy.enforce_security_policy` is unchanged; the guard check is now
  also available on its own as `enforce_guard_policy` for exactly this split.

### Fixed

- Add PEP 561 `py.typed` marker in `kuhaku` and declare it in `[tool.setuptools.package-data]`.
- Declare `prometheus-client>=0.20` as a direct runtime dependency in `pyproject.toml`.
- Fix `core/auth/policy.py`'s module docstring, which claimed `RAGEngine` has an
  `authorization_policy` constructor parameter; it never did.
- Remove leftover comments in four modules pointing at `service.build_service()`, a
  composition-root function that does not exist in this package.
- Fix CI: `tests/llm/test_vertex_provider.py` and `tests/rag/test_vertex_embeddings.py`
  import `google.genai` unconditionally, but CI's `pip install -e ".[dev]"` never
  installed the optional `vertex` extra -- every push to `main` has been failing on 18
  failed / 19 errored tests since the CI workflow was added. Both files now
  `pytest.importorskip("google.genai")`, matching how the rest of the suite treats
  optional dependencies (e.g. the cross-encoder re-ranker).

## [0.1.0] — 2026-08-25

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
