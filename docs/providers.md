# Providers

kuhaku separates two model choices: the **language model** that writes the answer, and
the **embedding model** that turns text into vectors for retrieval. They are configured
independently and have different consequences for what runs on your machine.

## Language models

Four providers ship with kuhaku. Select one with `KUHAKU_LLM_PROVIDER`.

| Provider | Value | Credentials | Default model |
|---|---|---|---|
| Ollama | `ollama` *(default)* | none — a local server | `qwen2.5:7b-instruct` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5` |
| Google Vertex AI | `vertex` | Application Default Credentials | `gemini-2.5-flash` |

### Ollama

The default. Nothing is sent anywhere, and no API key exists to leak.

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

```bash
export KUHAKU_OLLAMA_MODEL=llama3.1:8b
export KUHAKU_OLLAMA_BASE_URL=http://localhost:11434
```

If the server is not running, the first `ask()` fails with a message naming the two
commands above.

### OpenAI

```bash
export KUHAKU_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export KUHAKU_OPENAI_MODEL=gpt-4o-mini
```

`KUHAKU_OPENAI_BASE_URL` points the same client at any OpenAI-compatible server — a local
vLLM, an inference gateway, a proxy.

### Anthropic

```bash
export KUHAKU_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export KUHAKU_ANTHROPIC_MODEL=claude-sonnet-5
```

The endpoint is fixed and cannot be redirected.

### Vertex AI

Needs the optional extra and a Google Cloud project. Authentication is Application
Default Credentials — a `gcloud` login or a service account — not an API key field.

```bash
pip install "kuhaku[vertex]"
export KUHAKU_LLM_PROVIDER=vertex
export GOOGLE_CLOUD_PROJECT=my-project
export GOOGLE_CLOUD_LOCATION=us-central1
```

**`KUHAKU_LLM_TEMPERATURE` and `KUHAKU_LLM_MAX_TOKENS` have no effect on this provider.**
They apply to the other three.

### Configuring in code

```python
from kuhaku import RAG, Settings

rag = RAG(
    vector_store="./kuhaku-data",
    settings=Settings(
        llm_provider="openai",
        openai_api_key="sk-...",
        openai_model="gpt-4o-mini",
        llm_temperature=0.0,
    ),
)
```

`RAG(...)` has no provider or model argument of its own — provider selection goes through
the environment or a `Settings` object.

## Embedding models

Two backends. This is the choice that decides whether anything downloads.

| Backend | Value | Runs | Requires |
|---|---|---|---|
| sentence-transformers | `sentence-transformer` *(default)* | locally, on your CPU | ~490 MB model download |
| Vertex AI | `vertex` | Google's servers | `kuhaku[vertex]` + a GCP project |

**There is no OpenAI or Anthropic embedding backend.** Setting an API key for the
language model does not move embeddings off your machine — with `openai` or `anthropic`
selected, the local embedding model still downloads and still runs.

The only fully hosted configuration is Vertex for both halves:

```bash
pip install "kuhaku[vertex]"
export KUHAKU_LLM_PROVIDER=vertex
export KUHAKU_RAG__EMBEDDING_PROVIDER=vertex
export GOOGLE_CLOUD_PROJECT=my-project
```

Note the double underscore: embedding settings live on `RAGSettings`, so they nest under
`KUHAKU_RAG__`.

### Choosing a local model

```bash
export KUHAKU_RAG__EMBEDDING_MODEL=intfloat/multilingual-e5-base
export KUHAKU_RAG__EMBEDDING_DEVICE=cuda
```

Any sentence-transformers model works. Models with `e5` in the name automatically get the
`query: ` and `passage: ` prefixes those models expect.

Changing the embedding model invalidates an existing vector store — the old vectors were
produced by a different model and are not comparable. Use a new `vector_store` directory
or re-ingest.

## Re-ranking

Off by default, because the model is roughly 1.1 GB.

```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data", reranker=True)
```

```bash
export KUHAKU_RAG__RERANK_ENABLED=true
export KUHAKU_RAG__RERANKER_MODEL=BAAI/bge-reranker-base
export KUHAKU_RAG__RERANK_CANDIDATES=20
```

`reranker=` also takes a model name directly: `RAG(reranker="BAAI/bge-reranker-v2-m3")`.
Re-ranking applies to the sparse and hybrid paths, where a candidate pool exists to
re-order.

## Resilience

Every LLM call goes through a circuit breaker wrapped around a retry loop, bounded by a
timeout. Embedding, vector store and re-ranker calls have retries but no breaker.

| Setting | Default |
|---|---|
| `KUHAKU_RETRY_ENABLED` | `true` — master switch |
| `KUHAKU_RETRY_LLM_MAX_ATTEMPTS` | `3` |
| `KUHAKU_LLM_TIMEOUT_SECONDS` | `120` |
| `KUHAKU_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` consecutive failures open the circuit |
| `KUHAKU_CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS` | `60.0` before a probe is allowed |

Only server-side failures are retried: an HTTP 5xx or a connection error. A 4xx — a bad
key, a bad model name — fails immediately, because retrying it would only be slower.

When the circuit is open, calls fail immediately with an `LLMError` naming the remaining
cooldown, without touching the network.

## Writing your own provider

Any object with these two members is an `LLMProvider`:

<!-- no-exec -->
```python
class MyProvider:
    @property
    def name(self) -> str:
        return "myprovider:my-model"

    def generate(self, system: str, user: str) -> str:
        return call_my_api(system, user)
```

Pass it to `RAGEngine` directly — see [Extending](extending.md). If it also exposes a
`last_usage` attribute with `prompt_tokens` and `completion_tokens`, token metrics are
recorded automatically.
