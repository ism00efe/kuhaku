# kuhaku

[![CI](https://github.com/ism00efe/kuhaku/actions/workflows/ci.yml/badge.svg)](https://github.com/ism00efe/kuhaku/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kuhaku.svg)](https://pypi.org/project/kuhaku/)
[![Python](https://img.shields.io/pypi/pyversions/kuhaku.svg)](https://pypi.org/project/kuhaku/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

An AI orchestration framework for Python. `kuhaku.core` is tool-agnostic runtime
infrastructure — LLM abstraction, configuration, identity, security, observability,
retry. Tools are built on top of it, and retrieval-augmented generation is the first
one that ships.

> **Alpha — 0.1.0.**

---

## Why kuhaku

- **Access control that filters before it ranks.** Tag a document and a caller without a
  matching tag never retrieves it — in every retrieval strategy, before ranking, so an
  entitled caller still gets a full result set.
- **Evaluation ships with the framework.** A separate, tool-agnostic package with
  retrieval and answer-quality metrics that measures anything implementing the target
  contract.
- **No switch to leave in the wrong position.** PII sanitization has no flag at all, and
  access filtering has no global one — tagging a document is what protects it.

---

## Quickstart

```bash
pip install kuhaku
```

Point it at a hosted model — no local server needed:

```bash
export KUHAKU_LLM_PROVIDER=openai      # or: anthropic, vertex, ollama (the default)
export OPENAI_API_KEY=sk-...
```

```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data")
rag.ingest(open("handbook.md").read(), filename="handbook.md")

answer = rag.ask("How do I reset my password?")
print(answer.text)
```

`vector_store` names the directory the index lives in. Pass it: without one, kuhaku
creates a fresh temporary directory per `RAG()` instance, so anything you ingest is gone
the next time your program starts.

The answer text carries its citations inline as `[S1]`-style tags. `answer.citations`
maps each tag back to its document; `answer.retrieved`, `answer.redactions` (what PII
sanitization masked), `answer.abstained` and `answer.trace_id` are on the same object.

When retrieval finds nothing relevant, kuhaku abstains rather than letting the model
improvise.

Embeddings run locally whichever provider you choose, so the first ingest downloads a
~490 MB model — see [Installation in detail](#installation-in-detail) before you install
on a small disk.

---

## What it does today

- **Ingestion** — `.txt`, `.md`, `.pdf`; paragraph or structural chunking
- **Retrieval** — dense (embeddings), sparse (BM25), or both fused with Reciprocal Rank
  Fusion; optional cross-encoder re-ranking
- **Document-level access filtering** — flat tag intersection, enforced before ranking
- **Four LLM providers** — Ollama, OpenAI, Anthropic, Google Vertex AI
- **Security** — PII sanitization, deterministic prompt-injection input guard,
  per-request audit record
- **Observability** — structured logging with trace-id propagation, OpenTelemetry
  tracing and metrics
- **Resilience** — retry with exponential backoff and circuit breakers on every external
  call
- **Evaluation** — golden-dataset loader, retrieval metrics (hit rate@k, MRR, nDCG@k,
  precision, recall) and answer-quality metrics, in-memory or SQLite result stores
- **Caching** — query-answer cache keyed by the entitled chunk set, so two entitlements
  never share an entry

1065 tests, no failures, entirely offline — in-memory fakes for the embedder, vector
store and LLM. No network, no model download, no running LLM server needed to run them.

---

## Access control

```python
from kuhaku import RAG, AuthContext

rag = RAG()

rag.ingest(hr_text,  filename="salary_policy.md", access_tags=["people_ops"])
rag.ingest(eng_text, filename="deploy_runbook.md")          # untagged → visible to all

answer = rag.ask(
    "what are the salary bands",
    auth_context=AuthContext(identity="ada", roles=("engineering",)),
)
# salary_policy.md is never retrieved, never cited, never reaches the model
```

How it behaves:

- An **untagged** chunk is visible to everyone. Tagging is opt-in, so an existing corpus
  keeps working unchanged.
- A **tagged** chunk is visible only when one of its tags appears in the caller's `roles`.
  Flat set intersection — no hierarchy, no ordering, no tag implying another.
- A tagged chunk with **no `auth_context`** is not retrievable. Tagging a document is what
  turns protection on for it, which is why there is no enable/disable switch to forget.
- Filtering happens **before ranking**, in dense, sparse and hybrid alike.
- A result withheld for lack of entitlement is **indistinguishable from a genuine
  no-match**. Saying "you may not see this" would confirm that matching restricted
  material exists.

Tags are your vocabulary — kuhaku assigns no meaning to a tag string. `["people_ops"]`,
`["level-3"]` and `["muhasebe"]` behave identically. Three constants ship as a starting
point and nothing more:

```python
from kuhaku import ACCESS_TAG_PUBLIC, ACCESS_TAG_INTERNAL, ACCESS_TAG_RESTRICTED
```

**kuhaku does not authenticate anyone.** Your application proves who the user is and
hands kuhaku the result. kuhaku's job is enforcement: making sure retrieval cannot return
a chunk that `AuthContext` is not entitled to see.

---

## Installation in detail

Requires Python 3.11+.

### What `pip install kuhaku` pulls in

kuhaku depends on `chromadb`, `sentence-transformers` and `scikit-learn`, which together
resolve to roughly 120 packages. On Linux, about 19 of those are NVIDIA CUDA runtime
libraries (`nvidia-cublas`, `nvidia-cudnn`, `nccl`, `triton`, …) that PyTorch's default
wheel depends on — several gigabytes that are dead weight without an NVIDIA GPU.
Installing the CPU-only PyTorch build first avoids them entirely:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install kuhaku
```

Windows and macOS wheels do not pull separate CUDA packages, so this step is a Linux
concern.

### Models downloaded at runtime

| Model | When | Size |
|---|---|---|
| `intfloat/multilingual-e5-small` (embeddings) | first ingest | ~490 MB |
| `qwen2.5:7b-instruct` (LLM) | `ollama pull`, local route only | 4.7 GB |
| `BAAI/bge-reranker-base` (re-ranker) | only if you enable it | ~1.1 GB |

An API key removes the LLM download, not the embedding one. The only fully hosted
configuration is Vertex AI for both the LLM and the embeddings
(`KUHAKU_RAG__EMBEDDING_PROVIDER=vertex`), which needs a Google Cloud project and the
`vertex` extra:

```bash
pip install "kuhaku[vertex]"   # only if you are using Google Vertex AI
pip install "kuhaku[dev]"      # pytest, ruff, mypy, build
```

### Running the LLM locally

The default provider is a local [Ollama](https://ollama.com) server:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct   # or any other Ollama model you prefer
```

Point kuhaku at a different one with `KUHAKU_OLLAMA_MODEL`.

---

## Defaults

A bare `RAG()` configures nothing and downloads no model. The governing rule: **a default
may cost CPU and memory, never a download.**

| | Default |
|---|---|
| Retrieval | Hybrid — dense embeddings + BM25, fused with RRF |
| LLM | Ollama, `qwen2.5:7b-instruct`, `http://localhost:11434` |
| Embeddings | `intfloat/multilingual-e5-small`, CPU |
| Vector store | Chroma, collection `default_kb`, temporary directory |
| Chunking | Paragraph, 500 characters, 80 overlap |
| `top_k` | 4 |
| PII sanitization | Always on — no setting disables it |
| Prompt-injection input guard | Always on — the `RAG` facade exposes no way to disable it |
| Audit log | On — one record per request, whatever the outcome (`RAG(audit_enabled=False)` disables it) |
| Query cache | On — SQLite, 1 hour TTL |
| Cross-encoder re-ranker | **Off** — `BAAI/bge-reranker-base` is ~1.1 GB |

Every setting reads from the environment under a `KUHAKU_` prefix, with RAG settings
nested under `KUHAKU_RAG__`:

```bash
KUHAKU_RAG__TOP_K=8
KUHAKU_RAG__RETRIEVAL=dense
KUHAKU_RAG__RERANK_ENABLED=true
```

Four ecosystem-standard names are also accepted unprefixed, because another library
reading them means the same thing: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`.

kuhaku never configures logging or reads a `.env` file on its own — both belong to your
application.

Anything the `RAG` facade does not expose is reachable through `rag.engine`, a
`RAGEngine` built by constructor injection: swap the retriever, inject an authorization
policy, supply your own messages.

---

## Architecture

```
kuhaku/
├── core/          tool-agnostic runtime infrastructure
│   ├── config          typed, environment-driven Settings
│   ├── llm             LLMProvider + Ollama / OpenAI / Anthropic / Vertex AI
│   ├── auth            AuthContext and identity primitives
│   ├── security        prompt-injection input guard, PII sanitization, audit
│   ├── observability   structured logging, OpenTelemetry tracing and metrics
│   └── retry           retry with backoff, circuit breakers
├── evaluation/    tool-agnostic evaluation harness
└── tools/
    └── rag/       ingestion, chunking, embeddings, vector store, retrieval, fusion,
                   re-ranking, access filtering, caching
```

**The rule:** a tool may depend on `kuhaku.core`. Core may never depend on a tool. A
second tool is a new package under `tools/`, never a new field on core.

---

## Limitations

- `load_documents` applies one tag set to the whole call, not per file.
- The BM25 index rebuilds on the first query after an ingest rather than updating
  incrementally, so that query pays an O(corpus size) cost. `RAG(retrieval="dense")`
  avoids it.
- Chroma is the only vector store implementation, and there are no async APIs.

---

## Contributing

Issues and pull requests are welcome. The one hard requirement is architectural: `core`
stays tool-agnostic. See [CONTRIBUTING.md](https://github.com/ism00efe/kuhaku/blob/main/CONTRIBUTING.md).

```bash
git clone https://github.com/ism00efe/kuhaku
cd kuhaku
pip install -e ".[dev]"
pytest
```

## License

Apache License 2.0 — see [LICENSE](https://github.com/ism00efe/kuhaku/blob/main/LICENSE) and [NOTICE](https://github.com/ism00efe/kuhaku/blob/main/NOTICE).
