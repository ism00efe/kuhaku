# kuhaku

[![PyPI](https://img.shields.io/pypi/v/kuhaku.svg)](https://pypi.org/project/kuhaku/)
[![Python](https://img.shields.io/pypi/pyversions/kuhaku.svg)](https://pypi.org/project/kuhaku/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A provider-agnostic AI orchestration framework for Python. Retrieval-augmented generation
in three lines, with document-level access control, an evaluation harness and an audit
trail already in the box.

```python
from kuhaku import RAG

rag = RAG()
rag.ingest(open("handbook.md").read(), filename="handbook.md")
print(rag.ask("How do I reset my password?").text)
```

That runs against a local Ollama model. No API key, nothing sent anywhere.

> **Alpha.** Version 0.1.0. The API may change before 1.0. See
> [Status](#status) for what is and is not implemented.

---

## Why kuhaku

**RAG is a tool here, not the product.** `kuhaku.core` is tool-agnostic runtime
infrastructure — LLM abstraction, configuration, security, observability, retry. RAG is
the first tool built on top of it. Core never depends on a tool, so a second tool is a new
package under `tools/`, not a new field on core.

**Document-level access control that actually filters.** Tag a document, and a caller
without a matching tag never retrieves it — enforced *before* ranking, in every retrieval
strategy, so an entitled user still gets a full result set. Most implementations rank
first and drop afterwards, which silently returns nothing to someone who was allowed to
see material sitting just below the cut.

**Evaluation is not an afterthought.** `kuhaku.evaluation` is a tool-agnostic harness with
retrieval metrics (hit rate@k, MRR, nDCG@k, precision, recall) and answer-quality metrics,
runnable against `RAGEngine` or anything else implementing the target contract.

**Zero configuration, honest defaults.** Every knob is optional. The rule for what runs by
default: **a default may cost CPU and memory, never a download.** Hybrid retrieval is on
because it costs neither; the cross-encoder re-ranker is off because it is a gigabyte.

---

## Installation

Requires Python 3.11+.

```bash
pip install kuhaku
```

Optional extras:

```bash
pip install "kuhaku[vertex]"   # Google Vertex AI provider
pip install "kuhaku[dev]"      # pytest, ruff, mypy, build
```

### Prerequisites

The default provider is a local [Ollama](https://ollama.com) server:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

Prefer a hosted model instead? Set `KUHAKU_LLM_PROVIDER=openai` (or `anthropic`, `vertex`)
and the matching API key — no Ollama needed.

### What gets downloaded

| What | When | Approximate size |
|---|---|---|
| PyTorch (a `sentence-transformers` dependency) | on install | 2–3 GB |
| Embedding model `intfloat/multilingual-e5-small` | first ingest | ~0.5 GB |
| LLM `qwen2.5:7b-instruct` | `ollama pull` | ~4.7 GB |
| Re-ranker `BAAI/bge-reranker-base` | only if you enable it | ~1 GB |

Sizes are approximate and depend on your platform. Only the last one is optional at
runtime — nothing downloads a re-ranker unless you ask for it.

---

## Access control

The part worth reading even if you skim the rest.

```python
from kuhaku import RAG, AuthContext

rag = RAG()

rag.ingest(hr_text, filename="salary_policy.md", access_tags=["people_ops"])
rag.ingest(eng_text, filename="deploy_runbook.md")          # untagged → visible to all

answer = rag.ask(
    "what are the salary bands",
    auth_context=AuthContext(identity="ada", roles=("engineering",)),
)
# salary_policy.md is never retrieved, never cited, never reaches the model
```

**How it behaves**

- An **untagged** chunk is visible to everyone. Tagging is opt-in; existing corpora keep
  working.
- A **tagged** chunk is visible only when one of its tags appears in the caller's
  `roles`. Flat set intersection — no hierarchy, no ordering, no tag implying another.
- A tagged chunk with **no `auth_context`** is not retrievable. Tagging a document is what
  turns protection on for it, so there is no enable/disable switch to forget.
- Filtering happens **before ranking** in dense, sparse and hybrid alike.
- An unentitled empty result is indistinguishable from a genuine no-match. Saying
  "you may not see this" would confirm that matching restricted material exists.

**Tags are your vocabulary.** kuhaku never assigns meaning to a tag string —
`["people_ops"]`, `["level-3"]`, `["muhasebe"]` all work identically. Three constants ship
as a starting point and nothing more:

```python
from kuhaku import ACCESS_TAG_PUBLIC, ACCESS_TAG_INTERNAL, ACCESS_TAG_RESTRICTED
```

**What kuhaku does not do:** authentication. Your application proves who the user is and
hands kuhaku the result. kuhaku's job is enforcement — making sure retrieval cannot return
a chunk that `AuthContext` is not entitled to see.

---

## What you get without configuring anything

| | Default |
|---|---|
| Retrieval | Hybrid — dense embeddings + BM25, fused with Reciprocal Rank Fusion |
| LLM | Ollama, `qwen2.5:7b-instruct`, `http://localhost:11434` |
| Embeddings | `intfloat/multilingual-e5-small` on CPU |
| Vector store | ChromaDB, persistent |
| Chunking | Paragraph, 500 chars, 80 overlap |
| `top_k` | 4 |
| PII sanitization | On — email, token, IP, card, national ID, phone |
| Prompt-injection guard | On — the deterministic input guard |
| Audit log | On — one record per request, whatever the outcome |
| Query cache | On — SQLite, 1 hour TTL, keyed by the entitled chunk set |
| Re-ranker | **Off** — enable with `RAG(reranker=True)` |

The first query after ingestion builds the BM25 index over the whole store. With a large
corpus that is a visible one-off cost; `RAG(retrieval="dense")` avoids it.

---

## Going further

```python
rag = RAG(
    retrieval="hybrid",                          # "dense" | "sparse" | "hybrid"
    reranker=True,                               # ~1 GB model, significant VRAM
    embedding="intfloat/multilingual-e5-small",
    vector_store="./data/chroma",                # persistent directory
    cache=False,                                 # or a path to a cache database
    persona="You are a support engineer for a payments platform.",
    language_policy="Always answer in Turkish.",
)
```

`persona` and `language_policy` each override one layer of the system prompt. The
safety core underneath — instruction precedence, `[DOC]` data marking, a canary rule,
grounding, mandatory citations, contradiction handling — is framework-owned and applies
unconditionally. `system_prompt=` replaces the whole thing and hands that responsibility
back to you.

Anything the facade does not expose is reachable through `rag.engine`, a `RAGEngine` built
by constructor injection — swap the retriever, inject an authorization policy, supply your
own `EngineMessages`.

### The answer

```python
answer = rag.ask("...")
answer.text          # the generated answer, with [S1]-style citation tags
answer.citations     # [Citation(tag, document_id, title, doc_type, source_path, score)]
answer.retrieved     # the chunks it was grounded in
answer.redactions    # what PII sanitization masked, e.g. ["EMAIL×2"]
answer.trace_id      # correlates logs, metrics and the audit record
```

When retrieval finds nothing relevant, kuhaku abstains rather than letting the model
improvise.

---

## Configuration

Every setting reads from the environment with a `KUHAKU_` prefix. RAG-specific settings
nest under `KUHAKU_RAG__`:

```bash
KUHAKU_LLM_PROVIDER=openai
KUHAKU_RAG__TOP_K=8
KUHAKU_RAG__RETRIEVAL=dense
KUHAKU_RAG__RERANK_ENABLED=true
```

Four ecosystem-standard names are also accepted unprefixed, because another library
reading them means the same thing: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`. Everything else is kuhaku's own
vocabulary and carries the prefix.

Or construct settings directly:

```python
from kuhaku import RAG, Settings

rag = RAG(settings=Settings(llm_provider="openai", openai_api_key="sk-..."))
```

kuhaku never configures logging or reads a `.env` file on its own — both belong to your
application.

---

## Architecture

```
kuhaku/
├── core/          tool-agnostic runtime infrastructure
│   ├── config       typed, environment-driven Settings
│   ├── llm          LLMProvider + Ollama / Anthropic / OpenAI / Vertex AI
│   ├── auth         AuthContext, AuthorizationPolicy, API-key and JWT providers
│   ├── security     prompt-injection guard, output checks, PII sanitization, audit
│   ├── observability structured logging, OpenTelemetry tracing and metrics
│   └── retry        retry with backoff, circuit breakers
├── evaluation/    tool-agnostic evaluation harness
└── tools/
    └── rag/       the first tool: ingestion, chunking, embeddings, vector store,
                   retrieval, fusion, re-ranking, access filtering, caching
```

**The rule:** a tool may depend on `kuhaku.core`. Core may never depend on a tool. Adding
a second tool means a new `tools/<name>` package, never a tool-specific field on core.

---

## Evaluation

```python
from kuhaku.evaluation import EvaluationRunner, HitRateAtKMetric, MRRMetric

runner = EvaluationRunner(
    metrics=[HitRateAtKMetric(k=5), MRRMetric()],
    dataset_path="golden.jsonl",
)
scores = runner.run(rag.engine, top_k=5)
```

Retrieval metrics need no LLM. Answer-quality metrics (faithfulness, correctness) use an
LLM judge you supply via `judge_llm_provider`. Results go to an in-memory or SQLite store.

---

## Status

**Working and reachable from the public API**

Ingestion (`.txt`, `.md`, `.pdf`) · paragraph and structural chunking · dense, sparse and
hybrid retrieval · cross-encoder re-ranking · ChromaDB storage · document-level access
filtering · query-answer caching · PII sanitization · prompt-injection input guard ·
per-request audit logging · OpenTelemetry tracing and metrics · retry and circuit breakers
· four LLM providers · the evaluation harness.

1023 tests, all passing, entirely offline — in-memory fakes for the embedder, vector store
and LLM. No network, no model download, no running LLM server.

**In the package but not wired to the facade**

Query rewriting, contradiction detection, and the layered prompt-injection guard v2. Guard
v2 additionally needs a model you train yourself; no weights ship with kuhaku. These are
reachable through `rag.engine` if you construct `RAGEngine` directly, and are not
recommended for 0.1.0.

**Not implemented**

Per-file access tags in `load_documents` (one tag set applies to the whole call) ·
incremental BM25 updates (the index rebuilds on the next query after ingestion) ·
async APIs · vector stores other than Chroma.

---

## Contributing

Issues and pull requests are welcome. The one hard requirement is architectural: `core`
stays tool-agnostic. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/ism00efe/kuhaku
cd kuhaku
pip install -e ".[dev]"
pytest
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
