# Architecture

```
kuhaku/
├── core/          tool-agnostic runtime infrastructure
│   ├── config          typed, environment-driven Settings
│   ├── llm              LLMProvider + Ollama / OpenAI / Anthropic / Vertex AI
│   ├── auth             AuthContext, AuthorizationPolicy, API-key and JWT providers
│   ├── security          prompt-injection guard, output checks, PII sanitization, audit
│   ├── observability      structured logging, OpenTelemetry tracing and metrics
│   └── retry             retry with backoff, circuit breakers
├── evaluation/    tool-agnostic evaluation harness
└── tools/
    └── rag/       ingestion, chunking, embeddings, vector store, retrieval, fusion,
                   re-ranking, access filtering, caching
```

**The one hard rule:** a tool may depend on `kuhaku.core`. Core may never depend on a
tool. A second tool built on kuhaku is a new package under `tools/`, never a new branch
inside core.

## Why this split

`kuhaku.core` has no notion of a chunk, a retriever or an embedding — those are RAG
vocabulary, and RAG is one tool built on top of `core`. A future tool (a code-search
agent, a structured-extraction pipeline) reuses `core`'s LLM abstraction, auth,
security and observability without carrying any RAG-shaped baggage, and `core` itself
never grows a RAG-specific field to accommodate it.

`kuhaku.evaluation` sits beside `tools/`, not under it, for the same reason: its
`EvaluationTarget` contract (`evaluate_sample(query) -> EvaluationSample`) has no
RAG-specific type in it, so it can score a bare retriever, a classifier, or any other
target that satisfies the contract — `RAGEngine` is just the first thing that does.

## The `RAG` facade

`kuhaku.RAG` is a thin, opinionated wrapper around `RAGEngine` — see
[configuration](configuration.md) for its knobs and the `rag.engine` escape hatch into
`RAGEngine`'s full constructor-injection surface (custom retrievers, query rewriting,
contradiction detection, a custom `EngineMessages`, ...).

## Known limitations

- `load_documents` applies one tag set to the whole call, not per file.
- The BM25 index rebuilds on the first query after an ingest rather than updating
  incrementally, so that query pays an O(corpus size) cost. `RAG(retrieval="dense")`
  avoids it.
- Chroma is the only vector store implementation, and there are no async APIs.
