# kuhaku

[![CI](https://github.com/ism00efe/kuhaku/actions/workflows/ci.yml/badge.svg)](https://github.com/ism00efe/kuhaku/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kuhaku.svg)](https://pypi.org/project/kuhaku/)
[![Python](https://img.shields.io/pypi/pyversions/kuhaku.svg)](https://pypi.org/project/kuhaku/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

An AI orchestration framework for Python. `kuhaku.core` is tool-agnostic runtime
infrastructure — LLM abstraction, configuration, identity, security, observability,
retry. Tools are built on top of it. Two ship today: retrieval-augmented generation and
structured extraction.

> **Alpha — 0.2.0.**

---

## Why kuhaku

- **Access control that filters before it ranks.** Tag a document and a caller without a
  matching tag never retrieves it — in every retrieval strategy, before ranking, so an
  entitled caller still gets a full result set.
- **Evaluation ships with the framework.** A separate, tool-agnostic package with
  retrieval and answer-quality metrics that measures anything implementing the target
  contract.
- **Structured extraction that asks for what is missing.** Name the fields and kuhaku
  pulls them out of free text. A value the text lacks is never invented: the result says
  which one is missing and gives you a question to ask for it.
- **Start light.** `pip install kuhaku` pulls nothing heavy, and no default ever downloads
  a model. A feature's dependencies come when you use it.
- **No switch to leave in the wrong position.** In RAG, PII sanitization has no flag at
  all, and access filtering has no global one — tagging a document is what protects it.

---

## Quickstart

```bash
pip install kuhaku
```

Point it at a hosted model — no local server needed. Set a key and kuhaku uses it:

```bash
export OPENROUTER_API_KEY=sk-or-...    # free models available; or OPENAI_API_KEY, ...
```

### Structured extraction

```python
from kuhaku import Extractor

order = Extractor({"product": str, "quantity": float})

result = order.extract("tea please")
print(result.status)    # NEEDS_CLARIFICATION
print(result.question)  # Could you also tell me: quantity?  -- send it to the user

result = order.extract("3 boxes", prior=result)  # their reply completes the same order
print(result.status)    # COMPLETE
print(result.data)      # {'product': 'tea', 'quantity': 3}
```

That is the whole conversation loop: kuhaku keeps no state, so pass the previous result
back as `prior=` with each new message, and send `result.question` while there is one.
A field kuhaku should ask for must be required — one that accepts `None` (`str | None`)
is optional and never asked for. To tell orders from complaints or small talk, give it a
list of extractors, one per kind: `Extractor([order, complaint])` picks the right one for
each message.

See [Structured extraction](https://github.com/ism00efe/kuhaku/blob/main/docs/extraction.md)
for routing between schemas, multi-turn slot filling, evidence checks and your own rules.

### Retrieval-augmented generation

RAG needs a vector store and an embedding library, so it has its own extra:

```bash
pip install "kuhaku[rag]"
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

Embeddings run locally whichever provider you choose. Dense retrieval needs a ~490 MB
model: kuhaku asks before downloading it, and until it is on disk retrieval uses keyword
(BM25) search alone — see [Installation in detail](#installation-in-detail).

---

## What it does today

- **Structured extraction** — text to validated fields from a plain field list (or a Pydantic model);
  choosing between several kinds of message, multi-turn slot filling, retries, fallback models, evidence
  checks, your own rules, optional PII masking and input guard
- **Ingestion** — `.txt`, `.md`, `.pdf`; paragraph or structural chunking
- **Retrieval** — dense (embeddings), sparse (BM25), or both fused with Reciprocal Rank
  Fusion; optional cross-encoder re-ranking
- **Document-level access filtering** — flat tag intersection, enforced before ranking
- **Six LLM providers** — Ollama, OpenAI, Anthropic, Google Vertex AI, Groq, OpenRouter
- **Security** — PII sanitization, deterministic prompt-injection input guard,
  per-request audit record
- **Observability** — structured logging with trace-id propagation, OpenTelemetry
  tracing and metrics
- **Resilience** — retry with exponential backoff and circuit breakers on every external
  call
- **Evaluation** — golden-dataset loader, retrieval metrics (hit rate@k, MRR, nDCG@k,
  precision, recall), answer-quality metrics and field-by-field structured-output
  metrics, in-memory or SQLite result stores
- **Caching** — query-answer cache keyed by the entitled chunk set, so two entitlements
  never share an entry

The test suite runs entirely offline — in-memory fakes for the embedder, vector store
and LLM. No network, no model download, no running LLM server needed to run it.

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

The base install is light: pydantic, requests, PDF and file-type detection,
OpenTelemetry and PyJWT. Nothing heavy comes with it. Each feature that needs more has
its own extra:

| Extra | Adds | For |
|---|---|---|
| `kuhaku[rag]` | `chromadb`, `sentence-transformers` (pulls PyTorch) | RAG's vector store and local embeddings |
| `kuhaku[dense]` | `sentence-transformers` | local embeddings alone |
| `kuhaku[guard]` | `scikit-learn`, `joblib`, `numpy` | the trained prompt-injection guard model |
| `kuhaku[otlp]` | the OTLP trace exporter (pulls `grpcio`) | sending traces to a collector |
| `kuhaku[vertex]` | `google-genai` | Google Vertex AI |

Without the extra, a feature that needs it either asks before installing it (RAG) or
stops and names the command to run (the others). Nothing is installed silently.

PyTorch's default Linux wheel pulls about 19 NVIDIA CUDA runtime libraries
(`nvidia-cublas`, `nvidia-cudnn`, `nccl`, `triton`, …) — several gigabytes that are dead
weight without an NVIDIA GPU. Installing the CPU-only build first avoids them:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "kuhaku[rag]"
```

Windows and macOS wheels do not pull separate CUDA packages, so this step is a Linux
concern.

### Models downloaded at runtime

| Model | When | Size |
|---|---|---|
| `intfloat/multilingual-e5-small` (embeddings) | dense retrieval, after you approve it | ~490 MB |
| `BAAI/bge-reranker-base` (re-ranker) | only if you enable it | ~1.1 GB |

An API key removes the LLM download, not the embedding one. The only fully hosted
configuration is Vertex AI for both the LLM and the embeddings
(`KUHAKU_RAG__EMBEDDING_PROVIDER=vertex`), which needs a Google Cloud project and the
`vertex` extra:

```bash
pip install "kuhaku[vertex]"   # only if you are using Google Vertex AI
pip install "kuhaku[dev]"      # pytest, ruff, mypy, build, and every optional package
```

### Running the LLM locally

With no provider set, kuhaku continues with the provider you used last; the first time,
it uses a provider whose API key is set, and without one a running local
[Ollama](https://ollama.com) server with a model you pulled:

```bash
ollama serve
ollama pull llama3.2   # any Ollama chat model
```

With several pulled, kuhaku asks which one once; `KUHAKU_OLLAMA_MODEL` names it instead.

---

## Defaults

A bare `RAG()` or `Extractor(...)` configures nothing and downloads no model. The
governing rule: **a default may cost CPU and memory, never a download.**

| | Default |
|---|---|
| Retrieval | `auto` — hybrid (dense + BM25/RRF) when an embedding backend builds, else sparse-only |
| LLM | `auto` — the provider used last; else a provider whose API key is set (OpenRouter's free model first); else a running Ollama, with a model you pulled |
| Embeddings | `intfloat/multilingual-e5-small`; device `auto` — CUDA / MPS / CPU |
| Vector store | Chroma, collection `default_kb`, temporary directory |
| Chunking | Paragraph, 500 characters, 80 overlap |
| `top_k` | 4 |
| PII sanitization (RAG) | Always on — no setting disables it |
| Prompt-injection input guard (RAG) | Always on — the `RAG` facade exposes no way to disable it |
| PII masking, input guard, audit (extraction) | Off — each gets in the way of ordinary use; one setting turns it on for every extraction |
| Audit log | On — one record per request, whatever the outcome (`RAG(audit_enabled=False)` disables it) |
| Query cache | On — SQLite, 1 hour TTL |
| Cross-encoder re-ranker | **Off** — `BAAI/bge-reranker-base` is ~1.1 GB |

`retrieval`, `llm_provider` and `embedding_device` ship as `auto`: resolved once at
construction from what is installed or reachable, downgrading toward fewer external
dependencies and announcing any downgrade on the terminal. An explicit value always wins,
and `KUHAKU_AUTO=false` disables the probing entirely — see
[docs/configuration.md](https://github.com/ism00efe/kuhaku/blob/main/docs/configuration.md#auto).

Every setting reads from the environment under a `KUHAKU_` prefix. Each tool's settings
carry its own name — `KUHAKU_RAG__` for RAG, `KUHAKU_EXTRACTION__` for extraction:

```bash
KUHAKU_RAG__TOP_K=8
KUHAKU_RAG__RETRIEVAL=dense
KUHAKU_EXTRACTION__MAX_RETRIES=2
```

Six ecosystem-standard names are also accepted unprefixed, because another library
reading them means the same thing: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`.

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
│   ├── llm             LLMProvider + Ollama / OpenAI / Anthropic / Vertex AI /
│   │                   Groq / OpenRouter
│   ├── resolve         capability resolution: what is usable now, asked before installing
│   ├── auth            AuthContext and identity primitives
│   ├── security        input policy, prompt-injection guard, PII sanitization, audit
│   ├── observability   structured logging, OpenTelemetry tracing and metrics
│   └── retry           retry with backoff, circuit breakers
├── evaluation/    tool-agnostic evaluation harness
└── tools/
    ├── rag/         ingestion, chunking, embeddings, vector store, retrieval, fusion,
    │                re-ranking, access filtering, caching
    └── extraction/  schemas, extraction pipeline, slot filling, routing, evidence
```

**The rule:** a tool may depend on `kuhaku.core`. Core may never depend on a tool. A new
tool is a new package under `tools/` with its own settings, never a new field on core.

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
