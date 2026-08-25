# Configuration

Every setting is optional. `RAG()` with no arguments works, and the rule for what runs by
default is: **a default may cost CPU and memory, never a download.**

There are three ways to configure kuhaku, in increasing order of precedence:

1. **Environment variables**, prefixed `KUHAKU_`. RAG-specific settings nest under
   `KUHAKU_RAG__`.
2. **A `Settings` object** passed as `RAG(settings=...)`.
3. **Constructor arguments** on `RAG(...)`, which override both for the handful of
   options they cover.

kuhaku never reads a `.env` file on its own and never configures logging — both belong to
your application. To load a `.env`, build the settings yourself:

```python
from kuhaku import RAG, Settings

rag = RAG(settings=Settings(_env_file=".env"))
```

## Constructor arguments

These are the options `RAG(...)` accepts directly. Every one defaults to `None`, meaning
"use the setting".

| Argument | Effect |
|---|---|
| `retrieval` | `"dense"`, `"sparse"` or `"hybrid"` |
| `reranker` | `True`, `False`, or a HuggingFace cross-encoder model name |
| `chunking` | `"paragraph"` or `"structural"` |
| `embedding` | an embedding model name |
| `vector_store` | a directory for the Chroma store |
| `cache` | `True`, `False`, or a path to a cache database |
| `audit_enabled` | `False` turns the audit log off |
| `audit_log_path` | where audit records are written |
| `persona` | replaces the system prompt's persona layer |
| `language_policy` | replaces the output-language layer |
| `system_prompt` | replaces the entire system prompt, safety core included |
| `settings` | a pre-built `Settings` |
| `rag_settings` | a pre-built `RAGSettings` |
| `enable_token_tracking` | `False` skips per-call token accounting |
| `vertex_project`, `vertex_location` | Google Cloud project and region |

Passing an unknown keyword raises `TypeError` rather than being ignored.

Two of these need a warning:

- **`system_prompt` replaces the safety core too.** Instruction precedence, data marking,
  the canary rule, grounding, mandatory citations and contradiction handling all live in
  the default prompt. Supply your own and none of them apply unless you write them
  yourself. Use `persona` and `language_policy` when you only want to change tone or
  output language.
- **`rag_settings` skips the projection step.** When you pass one, the `vertex_project`,
  `vertex_location` and retry-master values on your object are used as-is instead of
  being taken from `Settings`.

## Core settings

Set with `KUHAKU_` + the field name in upper case, e.g. `KUHAKU_LLM_PROVIDER=openai`.

| Setting | Default | Notes |
|---|---|---|
| `llm_provider` | `ollama` | `ollama`, `openai`, `anthropic`, `vertex` |
| `llm_temperature` | `0.1` | ignored by the `vertex` provider |
| `llm_max_tokens` | `1024` | ignored by the `vertex` provider |
| `llm_timeout_seconds` | `120` | per request, all providers |
| `ollama_base_url` | `http://localhost:11434` | |
| `ollama_model` | `qwen2.5:7b-instruct` | |
| `openai_api_key` | — | also accepted unprefixed as `OPENAI_API_KEY` |
| `openai_model` | `gpt-4o-mini` | |
| `openai_base_url` | `https://api.openai.com/v1` | point at an OpenAI-compatible server |
| `anthropic_api_key` | — | also accepted as `ANTHROPIC_API_KEY` |
| `anthropic_model` | `claude-sonnet-5` | |
| `vertex_project` | — | also accepted as `GOOGLE_CLOUD_PROJECT` |
| `vertex_location` | `us-central1` | also accepted as `GOOGLE_CLOUD_LOCATION` |
| `vertex_model` | `gemini-2.5-flash` | |
| `audit_enabled` | `True` | one record per request, whatever the outcome |
| `audit_log_path` | `./logs/kuhaku_audit.jsonl` | |
| `retry_enabled` | `True` | master switch for every retry site |
| `retry_llm_max_attempts` | `3` | |
| `retry_llm_backoff_base_seconds` | `1.0` | exponential |
| `retry_llm_backoff_max_seconds` | `10.0` | |
| `circuit_breaker_enabled` | `True` | LLM calls only |
| `circuit_breaker_failure_threshold` | `5` | consecutive failures before opening |
| `circuit_breaker_reset_timeout_seconds` | `60.0` | |
| `circuit_breaker_success_threshold` | `1` | probes needed to close again |

Four names are accepted **without** the `KUHAKU_` prefix, because another library reading
them means the same thing: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION`. Everything else is kuhaku's own vocabulary and carries the
prefix.

## RAG settings

Set with `KUHAKU_RAG__` + the field name, e.g. `KUHAKU_RAG__TOP_K=8`.

| Setting | Default | Notes |
|---|---|---|
| `retrieval` | `hybrid` | `dense`, `sparse`, `hybrid` |
| `top_k` | `4` | chunks passed to the model |
| `rag_confidence_threshold` | `0.15` | below this, kuhaku abstains |
| `chunking_strategy` | `paragraph` | or `structural` |
| `chunk_size` | `500` | characters |
| `chunk_overlap` | `80` | characters |
| `max_upload_bytes` | — | rejects larger documents at ingest |
| `doc_type_prefix_mapping` | `{}` | JSON: filename prefix to document type |
| `chroma_persist_dir` | — | **empty means a new temp directory each run** |
| `chroma_collection` | `default_kb` | |
| `embedding_provider` | `sentence-transformer` | or `vertex` |
| `embedding_model` | `intfloat/multilingual-e5-small` | |
| `embedding_device` | `cpu` | |
| `rerank_enabled` | `False` | the model is ~1.1 GB |
| `reranker_model` | `BAAI/bge-reranker-base` | |
| `rerank_candidates` | `20` | pool size before re-ranking |
| `max_chunks_per_document` | `2` | sparse and hybrid only |
| `rrf_k` | `60` | hybrid fusion constant |
| `bm25_k1` | `1.5` | sparse and hybrid only |
| `bm25_b` | `0.75` | sparse and hybrid only |
| `cache_enabled` | `True` | |
| `cache_ttl_seconds` | `3600` | |
| `cache_db_path` | `./data/kuhaku_qa_cache.sqlite3` | |
| `vertex_embedding_model` | `gemini-embedding-001` | `embedding_provider=vertex` only |
| `vertex_embedding_dimensions` | `768` | `embedding_provider=vertex` only |
| `retry_embedding_max_attempts` | `2` | `sentence-transformer` only |
| `retry_embedding_backoff_base_seconds` | `0.5` | |
| `retry_embedding_backoff_max_seconds` | `5.0` | |
| `retry_vectorstore_max_attempts` | `2` | |
| `retry_vectorstore_backoff_seconds` | `0.5` | fixed wait |
| `retry_reranker_max_attempts` | `2` | only when a re-ranker is built |
| `retry_reranker_backoff_base_seconds` | `1.0` | |
| `retry_reranker_backoff_max_seconds` | `5.0` | |

Three `KUHAKU_RAG__` names are **overwritten** on the ordinary path and will not take
effect: `KUHAKU_RAG__VERTEX_PROJECT`, `KUHAKU_RAG__VERTEX_LOCATION` and
`KUHAKU_RAG__RETRY_ENABLED`. Use the unprefixed core settings instead
(`KUHAKU_VERTEX_PROJECT`, `KUHAKU_VERTEX_LOCATION`, `KUHAKU_RETRY_ENABLED`), which feed
both layers.

## Storage kuhaku writes to disk

Three paths, all relative to the working directory unless you set them:

| What | Default location | Turn off with |
|---|---|---|
| Vector store | a new temp directory per instance | — set `vector_store=` instead |
| Query cache | `./data/kuhaku_qa_cache.sqlite3` | `RAG(cache=False)` |
| Audit log | `./logs/kuhaku_audit.jsonl` | `RAG(audit_enabled=False)` |

The cache file is created lazily, on the first request that reaches a cache lookup — a
`RAG()` that answers nothing creates no file. A cache that cannot be opened degrades to
no caching with a warning; it never fails a query. An audit log that cannot be written
also never fails a query, which means an unwritable path silently ends your audit
coverage — check for the warning on startup.

## Reading settings back

```python
from kuhaku import Settings, get_settings

settings = get_settings()
print(settings.llm_provider, settings.rag.top_k)

explicit = Settings(llm_provider="openai", openai_api_key="sk-...")
print(explicit.llm_provider)
```

`get_settings()` is cached on its arguments. Call `get_settings.cache_clear()` after
changing environment variables in the same process.

## Fields that exist but are not wired in 0.1.0

These appear on `Settings` and `RAGSettings` and can be set without error, but nothing
reads them on any path a `RAG()` caller can reach. They are listed here so that no one —
person or assistant — spends time setting them:

`input_guard_enabled`, `guard_enabled`, `guard_high_threshold`,
`guard_stage1_model_path`, `guard_stage2_onnx_path`, `guard_stage2_tokenizer_path`,
`guard_norm_drift_tolerance`, `guard_citation_grounding_threshold`,
`guard_model_version`, `guard_version`, `strict_performance_components`,
`metrics_enabled`, `llm_model_version`, `embedding_model_version`,
`prod_prompt_version`, `eval_prompt_version`, `contradiction_detection_enabled`,
`corpus_dir`.
