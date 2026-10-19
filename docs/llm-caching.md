# LLM caching

`CachingLLM` in `kuhaku.core.llm` wraps any `LLMProvider` and serves repeated identical
calls from a cache — in memory or in SQLite. It has the same `generate(system, user)` /
`name` shape, so anything that accepts a provider accepts it.

It is not switched on by a setting, and the `RAG(...)` facade does not accept a
provider, so a `RAG()` caller cannot reach it. Use them wherever you construct the
provider yourself: `RAGEngine(embedder, store, llm)` (see [Extending](extending.md)), the extraction
tool's `Extractor(llm=...)`, or an evaluation judge such as
`LLMFaithfulnessEvaluator(provider)`.

## Response cache

<!-- no-exec -->
```python
from kuhaku.core.llm import CachingLLM, SQLiteCacheBackend
from kuhaku.core.llm.groq_provider import GroqProvider

llm = CachingLLM(
    GroqProvider(api_key="...", model="llama-3.3-70b-versatile"),
    SQLiteCacheBackend("./data/llm-cache.sqlite3"),
    ttl_seconds=24 * 3600,
)
```

A self-contained version you can run as is:

```python
from kuhaku.core.llm import CachingLLM, InMemoryCacheBackend


class EchoProvider:
    calls = 0

    @property
    def name(self) -> str:
        return "echo:v1"

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return f"echo: {user}"


provider = EchoProvider()
llm = CachingLLM(provider, InMemoryCacheBackend(max_entries=1000))

llm.generate("You are terse.", "hello")
llm.generate("You are terse.", "hello")      # served from the cache
assert provider.calls == 1
print(llm.stats)                             # CacheStats(hits=1, misses=1)

llm.invalidate("You are terse.", "hello")    # drop one entry
llm.clear()                                  # drop everything in the backend
```

**The key** is a SHA-256 of the provider's `name` (provider and model), the generation
parameters that change the output (temperature, max tokens, and the base URL, read from
the bundled providers; pass `params={...}` for your own), the full system and user
prompt, and the optional namespace. Change any of them and the call misses. Provider
errors are never cached.

**Backends.** `InMemoryCacheBackend(max_entries=1024)` is a per-process LRU.
`SQLiteCacheBackend(path)` persists across runs and processes (stdlib `sqlite3`, WAL,
nothing written until the first call). Any object with `get`, `put`, `delete` and
`clear` is a backend — see `CacheBackend`. Both bundled backends are thread-safe, and a
failing backend degrades to a miss rather than failing the call.

**TTL.** `ttl_seconds=None` (the default) never expires entries. Expiry is checked on
read.

**Concurrency.** Two threads missing the same key at once each call the provider; the
cache does not coalesce them.

### Token accounting

On a hit the wrapper reports zero tokens through `last_usage`, and the wrapped provider's
real usage on a miss. Put it *inside* `TokenTrackingLLM` so the token metrics count what
was actually spent:

```python
from kuhaku.core.llm import CachingLLM
from kuhaku.core.llm.base import TokenUsage
from kuhaku.core.llm.token_tracking import TokenTrackingLLM


class MeteredProvider:
    last_usage = None

    @property
    def name(self) -> str:
        return "metered:v1"

    def generate(self, system: str, user: str) -> str:
        self.last_usage = TokenUsage(prompt_tokens=12, completion_tokens=3)
        return "ok"


llm = TokenTrackingLLM(CachingLLM(MeteredProvider()), provider="metered")
llm.generate("sys", "q")   # records 12 + 3 tokens
llm.generate("sys", "q")   # records 0 + 0
```

### Security: what a cache hit can reveal

A cached response is only served for a byte-identical prompt. Whatever a caller was
entitled to see — retrieved documents, a user's record — is part of the prompt text, so a
caller can only hit an entry if they already sent exactly that content. This is the same
guarantee the RAG answer cache gets by keying on the entitled chunk set.

What the prompt cannot express is a boundary you want enforced regardless of content —
tenants that must never share anything, even an answer to an identical generic question.
For that, set a namespace. It can be a string, or a callable evaluated on every call:

```python
from contextvars import ContextVar

from kuhaku.core.llm import CachingLLM, InMemoryCacheBackend


class EchoProvider:
    @property
    def name(self) -> str:
        return "echo:v1"

    def generate(self, system: str, user: str) -> str:
        return f"echo: {user}"


current_tenant: ContextVar[str] = ContextVar("current_tenant", default="public")
shared = InMemoryCacheBackend()
llm = CachingLLM(EchoProvider(), shared, namespace=current_tenant.get)

token = current_tenant.set("acme")
llm.generate("sys", "hello")       # cached under "acme" only
current_tenant.reset(token)
llm.generate("sys", "hello")       # "public": a miss
assert llm.stats.misses == 2
```

The SQLite file holds responses in plain text. Put it where prompt and response content
is allowed to rest.

## Caching other call shapes

`CachingLLM` is an `LLMWrapper`, so it presents the whole v2 surface. Besides
`generate` (`kind="generate"`), `chat`, `generate_structured` and `chat_with_tools` are
cached under the kinds `"chat"`, `"structured"` and
`"tools"`. Their key covers the messages (text verbatim; images, documents and tool
parts by their sha256 fingerprint, so bytes never enter a key), the per-call
`GenerationOptions`, and the JSON Schema or tool definitions. A hit returns the stored
response — tool calls included — with zero token usage.

`llm_cache_key(provider, request, kind=..., params=..., namespace=...)` is public and
takes any JSON-serialisable `request`, tagged with a `kind`; the key format does not
change between kinds.
