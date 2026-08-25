# kuhaku

An AI orchestration framework for Python. `kuhaku.core` is tool-agnostic runtime
infrastructure — LLM abstraction, configuration, identity, security, observability,
retry. Tools are built on top of it, and retrieval-augmented generation is the first one
that ships.

> **Alpha — 0.1.0.**

```bash
pip install kuhaku
```

```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data")
rag.ingest("Deploys run from the main branch every Tuesday.", filename="runbook.md")

answer = rag.ask("When do deploys run?")
print(answer.text)
```

## The pages

**[Getting started](getting-started.md)** — install, choose where the model runs, and get
your first answer. Start here.

**[Configuration](configuration.md)** — every setting that has an effect, its default,
and the environment variable that changes it. Also the fields that exist but are not
wired in this release.

**[Access control](access-control.md)** — restricting a document to the callers entitled
to see it, how the decision is made, and the two ways tags silently fail to match.

**[Security](security.md)** — PII sanitization, the prompt-injection input guard, the
layers of the system prompt, and the audit log.

**[Providers](providers.md)** — the four language models and the two embedding backends,
and which combination avoids a local download.

**[Observability](observability.md)** — trace ids, structured logging, OpenTelemetry
spans and the metrics kuhaku emits.

**[Extending](extending.md)** — replacing the retriever, store, model or user-facing
text with your own, and the contracts each one has to satisfy.

**[Evaluation](evaluation.md)** — the golden-dataset format, the retrieval and
answer-quality metrics, and how to measure something other than RAG.

## How these docs are written

Every code block is complete: the imports are there, and nothing depends on a variable
defined on another page. Copy any of them and it runs.

Where something does not work through the public `RAG` API, the page says so instead of
staying quiet — including for features the package contains. A control you can set that
changes nothing is worse than a missing one, because it reads as working.

## The architecture

```
kuhaku/
├── core/          tool-agnostic runtime infrastructure
│   ├── config          typed, environment-driven Settings
│   ├── llm             LLMProvider + Ollama / OpenAI / Anthropic / Vertex AI
│   ├── auth            AuthContext and identity primitives
│   ├── security        input guard, PII sanitization, audit
│   ├── observability   structured logging, OpenTelemetry tracing and metrics
│   └── retry           retry with backoff, circuit breakers
├── evaluation/    tool-agnostic evaluation harness
└── tools/
    └── rag/       ingestion, chunking, embeddings, vector store, retrieval, fusion,
                   re-ranking, access filtering, caching
```

**The rule:** a tool may depend on `kuhaku.core`. Core may never depend on a tool. A
second tool is a new package under `tools/`, never a new field on core.

## Links

- [Source](https://github.com/ism00efe/kuhaku)
- [Issues](https://github.com/ism00efe/kuhaku/issues)
- [Changelog](https://github.com/ism00efe/kuhaku/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/ism00efe/kuhaku/blob/main/CONTRIBUTING.md)
