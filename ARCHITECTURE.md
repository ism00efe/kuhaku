# Architecture

```
kuhaku/
├── core/          tool-agnostic runtime infrastructure
│   ├── config          typed, environment-driven Settings
│   ├── llm              LLMProvider + Ollama / OpenAI / Anthropic / Vertex AI / Groq /
│   │                    OpenRouter; with_runtime (every layer core applies to calls)
│   ├── resolve          capability resolution: probe → enumerate → branch on count;
│   │                    registry of adapters, UI/consent/decision-memory ports
│   ├── auth             AuthContext, AuthorizationPolicy, API-key and JWT providers
│   ├── security          input policy, prompt-injection guard, output checks,
│   │                    PII sanitization, audit
│   ├── observability      structured logging, OpenTelemetry tracing and metrics,
│   │                    traced_call (one tool call)
│   └── retry             retry with backoff, circuit breakers
├── evaluation/    tool-agnostic evaluation harness
└── tools/
    ├── rag/         ingestion, chunking, embeddings, vector store, retrieval, fusion,
    │                re-ranking, access filtering, caching
    └── extraction/  schemas, extraction pipeline, slot filling, routing, evidence
```

**The one hard rule:** a tool may depend on `kuhaku.core`. Core may never depend on a
tool. A new tool built on kuhaku is a new package under `tools/`, never a new branch
inside core. Each tool owns its settings (`RAGSettings`, `ExtractionSettings`), read from
its own `KUHAKU_<TOOL>__` variables; core's `Settings` carries none of them.

## Why this split

`kuhaku.core` has no notion of a chunk, a retriever or an embedding — those are RAG
vocabulary, and RAG is one tool built on top of `core`. Structured extraction, the second
tool, reuses `core`'s LLM abstraction, security and observability without carrying any
RAG-shaped baggage, and `core` never grew a field to accommodate either.

A tool reaches core through three entry points rather than wiring each piece itself —
`with_runtime` for model calls, `traced_call` for a call's observability and
`InputPolicy` for input checks — so what core adds later reaches every tool without a
change to the tool.

`kuhaku.evaluation` sits beside `tools/`, not under it, for the same reason: its
`EvaluationTarget` contract (`evaluate_sample(query) -> EvaluationSample`) has no
RAG-specific type in it, so it can score a bare retriever, a classifier, or any other
target that satisfies the contract — `RAGEngine` and `Extractor` both do.

## The `RAG` facade

`kuhaku.RAG` is a thin, opinionated wrapper around `RAGEngine` — see
[configuration](configuration.md) for its knobs and the `rag.engine` escape hatch into
`RAGEngine`'s full constructor-injection surface (custom retrievers, query rewriting,
contradiction detection, a custom `EngineMessages`, ...).

## Capability resolution

`core/resolve` decides which backend to use for each external dependency. Adapters
(under `core/resolve/adapters/` and `tools/rag/resolve/adapters/`) each report only
their own candidates and each candidate's `Cost`; the resolver branches on how many are
usable right now and never names a specific backend. Choosing an LLM looks only at
credentials and the Ollama endpoint, never the GPU, and continues with the provider used
last; a pinned provider is built directly. `RAG.__init__` and
`build_llm_provider` share one environment probe, one UI and one decision memory
(`.kuhaku/decisions.json`) across all their decisions.

The built-in lightweight (Tier-0) store and Qdrant are not implemented yet: Chroma is
the only store candidate, so the store decision always takes the "exactly one" branch,
and the "no reprocessing on a store upgrade" property is not yet achievable.

## Known limitations

- `load_documents` applies one tag set to the whole call, not per file.
- The BM25 index rebuilds on the first query after an ingest rather than updating
  incrementally, so that query pays an O(corpus size) cost. `RAG(retrieval="dense")`
  avoids it.
- Chroma is the only vector store implementation, and there are no async APIs.
