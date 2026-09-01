# Extending

`RAG` is a facade over `RAGEngine`. It builds the embedder, vector store, LLM, retriever,
chunker and cache from settings, then hands them to the engine. When you need a piece
kuhaku did not build, drop to the engine.

There are two ways in.

## Reaching the engine behind a facade

`rag.engine` returns the live `RAGEngine`. It has setters for most of its parts, so you
can keep the convenience of `RAG(...)` and replace one component:

<!-- no-exec -->
```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data")
rag.engine.update_retriever(MyRetriever())
answer = rag.ask("How do I reset my password?")
```

Available setters: `update_llm`, `update_embedder`, `update_vector_store`,
`update_retriever`, `update_chunker`, `update_query_rewriter`,
`update_contradiction_detector`, `update_system_prompt`, `update_top_k`,
`update_confidence_threshold`, `update_cache`, `update_guard`.

The engine also exposes `answer(question, context_text=None, trace_id=None, *,
auth_context=None)` — note the `trace_id`, which `RAG.ask` does not accept.

## Constructing the engine yourself

Full control, and the only way to reach the parameters the facade never passes:

<!-- no-exec -->
```python
from kuhaku.tools.rag.engine import RAGEngine

engine = RAGEngine(
    my_embedder,
    my_store,
    my_llm,
    top_k=4,
    retriever=my_retriever,
    chunker=my_chunker,
    confidence_threshold=0.15,
    audit_enabled=True,
    audit_log_path="./logs/kuhaku_audit.jsonl",
)

answer = engine.answer("How do I reset my password?", trace_id="req-1234")
```

The first three arguments are positional and required.

Parameters that exist only here, not on `RAG(...)`: `input_guard_enabled`, `guard`,
`query_rewriter`, `contradiction_detector`, `contradiction_db_path`,
`contradiction_storage`, `messages`, `llm_version`, `embedding_version`,
`system_prompt_version`.

## The contracts

Each of these is a protocol — implement the methods, no base class to inherit.

### Retriever

```python
class MyRetriever:
    def retrieve(self, query, top_k, *, auth_context=None, doc_type=None):
        ...  # -> list[RetrievedChunk]
```

An optional `refresh(self) -> None` is called after each ingest if you define it — use it
to rebuild an index.

**You own access control here.** The entitlement rule lives in the built-in retrievers,
not in the engine, so a custom retriever that ignores `auth_context` returns restricted
chunks to everyone. Apply the rule:

<!-- no-exec -->
```python
from kuhaku.tools.rag.retriever import is_entitled

class MyRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, top_k, *, auth_context=None, doc_type=None):
        hits = my_search(query, self._chunks)
        return [h for h in hits if is_entitled(h.chunk, auth_context)][:top_k]
```

### Embedding provider

```python
class MyEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...
```

### Vector store

```python
class MyStore:
    def add(self, chunks, embeddings) -> None: ...
    def query(self, embedding, top_k, *, where=None): ...   # -> list[RetrievedChunk]
    def count(self) -> int: ...
    def reset(self) -> None: ...
    def list_collections(self) -> list[str]: ...
    def iter_chunks(self): ...
```

`where` is a backend-native filter that must be applied **before** ranking. The dense
retriever uses it to push entitlement filtering into the store; a store that ignores
`where` breaks that guarantee.

### LLM provider

```python
class MyLLM:
    @property
    def name(self) -> str:
        return "myprovider:my-model"

    def generate(self, system: str, user: str) -> str:
        ...
```

To have `KUHAKU_LLM_PROVIDER=auto` consider a new backend, add an adapter under
`kuhaku/core/resolve/adapters/` that reports a `Candidate` for it (its readiness, its
`Cost`, and a zero-arg `activate` that builds it) and register it in
`register_llm_adapters`. The resolver itself never names a provider — it only counts
candidates and reads their cost.

### Chunker

```python
class MyChunker:
    def chunk(self, doc, *, chunk_size: int, overlap: int):
        ...  # -> list[Chunk]
```

### Cache

Anything with `get(key) -> str | None` and `put(key, text) -> None`.

### Reranker

```python
class MyReranker:
    def rerank(self, query: str, candidates, top_k: int):
        ...  # -> list[RetrievedChunk]
```

## Replacing user-visible text

Every message kuhaku shows a user — the abstention sentence, the empty-corpus notice,
file-handling errors — lives on one frozen dataclass, and all of them are English.
Replace it through the engine:

<!-- no-exec -->
```python
from dataclasses import replace
from kuhaku.tools.rag.engine import RAGEngine
from kuhaku.tools.rag.messages import DEFAULT_ENGINE_MESSAGES

messages = replace(
    DEFAULT_ENGINE_MESSAGES,
    no_chunks="Bu soruya cevap verecek bir kaynak bulamadım.",
    empty_kb="Bilgi tabanı boş.",
)

engine = RAGEngine(my_embedder, my_store, my_llm, messages=messages)
```

Some fields are templates with named placeholders — `document_security_check_failed_template`
takes `{reason}`, `upload_size_exceeded_template` takes `{max_bytes}`. Keep the
placeholders when you translate.

This is not reachable through `RAG(...)`.

## Adding a second tool

The architectural rule: **a tool may depend on `kuhaku.core`; core may never depend on a
tool.** A second tool is a new package under `kuhaku/tools/<name>/`, with its own
`<Name>Settings` dataclass and a field for it on `Settings` — never a tool-specific field
added directly to core.

`kuhaku.core` gives a new tool the LLM abstraction, typed configuration, `AuthContext`,
sanitization, the audit log, observability and retry. `kuhaku.evaluation` measures
anything that implements the evaluation target contract, not just RAG.

Three types are exported for exactly this purpose — `Message`, `ToolCall` and
`ExecutionResult`. They are a vocabulary offered to a second tool; nothing in the RAG
tool consumes them today.

## Not in 0.1.0

Query rewriting and contradiction detection ship in the package and can be passed to
`RAGEngine` directly, but are not wired to the facade and are not recommended for this
release. The second-generation guard needs a model you train yourself and has no factory
— see [Security](security.md).
