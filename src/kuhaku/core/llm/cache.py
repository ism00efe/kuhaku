"""Response-caching decorator around any :class:`~kuhaku.core.llm.base.LLMProvider`.

``CachingLLM`` has the same ``generate()``/``name`` shape as the provider it wraps, so a
caller (``RAGEngine``, ``Extractor``, ``LLMJudge``, ...) never knows it is there. The
storage is a pluggable :class:`CacheBackend`; :class:`InMemoryCacheBackend` (LRU) and
:class:`SQLiteCacheBackend` (stdlib ``sqlite3``) ship here.

The key is a SHA-256 over a canonical JSON document of everything that decides the
output: the provider's ``name`` (``"<provider>:<model>"``), the generation parameters
that affect sampling, the full request (system *and* user prompt), an optional
namespace, and a ``kind`` tag. ``kind`` is ``"generate"`` for ``generate(system,
user)`` -- and for the identical ``chat([system, user])`` with no options, which is
the same logical call; the other v2 calls are cached under their own
kinds -- ``"chat"``,
``"structured"`` and ``"tools"`` -- with a request of the messages (text verbatim,
any other content part by its fingerprint, so image bytes never enter a key), the
per-call options, and the schema or tool definitions.

SECURITY: compare ``tools/rag/cache.py``, which keys a RAG answer by the *entitled chunk
set* -- the query alone is not enough there because two callers with different access
tags asking the same question must not share an answer. Here that property falls out of
keying on the full prompt: whatever context a caller was entitled to is already in the
prompt text, so a hit means the caller sent byte-identical input and would have been
shown the same content anyway. What the prompt cannot express (a tenant boundary, a
policy that forbids even identical-input sharing) is what ``namespace`` is for.

Lookups and stores are best-effort, mirroring ``token_tracking.py``: a backend failure
is logged and treated as a miss / no-op -- a cache must never turn a working answer into
a failed request. Provider errors are never cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..models import Message, TextPart, ToolCall
from ..observability.metrics import record_llm_cache_lookup
from ..observability.tracing import set_current_span_attributes
from .base import (
    GenerationOptions,
    LLMProvider,
    LLMResponse,
    StructuredResponse,
    TokenUsage,
)
from .tool_calling import ToolSpec
from .wrapper import LLMWrapper, generate_call, generate_response

logger = logging.getLogger(__name__)

# Bumped if the canonical key document ever changes shape, so old persisted entries
# simply stop matching instead of being served for a request they were not made for.
KEY_VERSION = 1

# Span attributes set on whatever span is current when a lookup happens (typically the
# "generate" stage span). Not GenAI-convention keys -- there is no convention for caching.
SPAN_CACHE_HIT = "kuhaku.llm.cache.hit"
SPAN_CACHE_SOURCE = "kuhaku.llm.cache.source"

# The usage a hit reports, so a ``TokenTrackingLLM`` wrapped around this records zero
# tokens spent rather than nothing at all (a hit is a call that cost nothing, not an
# untracked call).
ZERO_USAGE = TokenUsage(prompt_tokens=0, completion_tokens=0)

# Private attributes the bundled providers keep their sampling knobs in. Read duck-typed
# (same idiom as ``last_usage``) so caching needs no change to any provider; a provider
# that keeps them elsewhere should be wrapped with an explicit ``params=``.
_INTROSPECTED_PARAMS = ("_temperature", "_max_tokens", "_base_url")


def generation_params(provider: object) -> dict[str, object]:
    """The output-affecting parameters a bundled provider was constructed with.

    ``base_url`` is included because an OpenAI-compatible endpoint can serve a different
    model under the same name; the cost of including it is only extra misses.

    Decorators are looked through: ``build_llm_provider`` hands out a
    ``RateLimitedLLM`` when rate limits are configured, and reading only the outermost
    object would give ``{}`` -- so a persistent cache would serve an answer sampled at
    another temperature or from another endpoint. The chain is followed through
    ``wrapped`` until an object that
    exposes its parameters: a public ``generation_params`` mapping, or the bundled
    providers' private attributes.
    """

    seen: set[int] = set()
    current: object | None = provider
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        explicit = getattr(current, "generation_params", None)
        if isinstance(explicit, Mapping):
            return dict(explicit)
        params: dict[str, object] = {}
        for attr in _INTROSPECTED_PARAMS:
            value = getattr(current, attr, None)
            if value is not None:
                params[attr.lstrip("_")] = value
        if params:
            return params
        inner = getattr(current, "wrapped", None)
        current = inner if callable(getattr(inner, "generate", None)) else None
    return {}


def llm_cache_key(
    provider: str,
    request: Mapping[str, object],
    *,
    kind: str = "generate",
    params: Mapping[str, object] | None = None,
    namespace: str | None = None,
) -> str:
    """SHA-256 hex digest identifying one LLM call.

    ``request`` and ``params`` must be JSON-serialisable; keys are sorted, so dict
    ordering never changes the key. Every field is a separate JSON member rather than a
    concatenated string, so no choice of prompt text can collide two distinct calls.
    """

    document = {
        "v": KEY_VERSION,
        "kind": kind,
        "provider": provider,
        "params": dict(params or {}),
        "namespace": namespace,
        "request": dict(request),
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def provider_system(provider_name: str) -> str:
    """``"openai:gpt-4o"`` -> ``"openai"``: a low-cardinality metric label."""

    return provider_name.split(":", 1)[0]


def observe_lookup(*, source: str, hit: bool, provider_name: str) -> None:
    """Best-effort metric + span annotation for one lookup; never raises."""

    try:
        record_llm_cache_lookup(
            source=source, result="hit" if hit else "miss", system=provider_system(provider_name)
        )
        set_current_span_attributes(**{SPAN_CACHE_HIT: hit, SPAN_CACHE_SOURCE: source})
    except Exception:
        logger.exception("llm cache observability failed (answer unaffected)")


# --- v2 call shapes ---------------------------------------------------------------------

V2Response = TypeVar("V2Response", LLMResponse, StructuredResponse)


def _message_request(message: Message) -> dict[str, object]:
    parts: list[object] = [
        part.text if isinstance(part, TextPart) else {"type": part.type, "sha256": part.fingerprint}
        for part in message.parts
    ]
    return {"role": message.role, "parts": parts}


def _options_request(options: GenerationOptions | None) -> dict[str, object] | None:
    if options is None:
        return None
    return {"temperature": options.temperature, "max_tokens": options.max_tokens}


def chat_request(
    messages: Sequence[Message], options: GenerationOptions | None
) -> dict[str, object]:
    """The ``request`` of a ``"chat"`` key."""

    return {
        "messages": [_message_request(m) for m in messages],
        "options": _options_request(options),
    }


def structured_request(
    messages: Sequence[Message],
    json_schema: dict[str, Any],
    name: str,
    options: GenerationOptions | None,
    max_repair_attempts: int,
) -> dict[str, object]:
    """The ``request`` of a ``"structured"`` key."""

    return {
        **chat_request(messages, options),
        "schema": json_schema,
        "name": name,
        "max_repair_attempts": max_repair_attempts,
    }


def tools_request(
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    tool_choice: str,
    options: GenerationOptions | None,
) -> dict[str, object]:
    """The ``request`` of a ``"tools"`` key."""

    return {
        **chat_request(messages, options),
        "tools": [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in tools
        ],
        "tool_choice": tool_choice,
    }


def encode_response(response: LLMResponse | StructuredResponse) -> str | None:
    """A v2 response as stored text (JSON), or ``None`` when it cannot be stored (its
    data or metadata is not JSON-serialisable) -- it is then simply not cached."""

    if isinstance(response, StructuredResponse):
        document: dict[str, object] = {
            "data": response.data,
            "text": response.text,
            "finish_reason": response.finish_reason,
            "native": response.native,
            "attempts": response.attempts,
        }
    else:
        document = {
            "text": response.text,
            "finish_reason": response.finish_reason,
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "id": call.id,
                    "raw_arguments": call.raw_arguments,
                    "arguments_error": call.arguments_error,
                }
                for call in response.tool_calls
            ],
            "metadata": response.metadata,
        }
    try:
        return json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return None


def decode_response(kind: str, text: str, model: str) -> LLMResponse | StructuredResponse:
    """The response :func:`encode_response` stored, reporting :data:`ZERO_USAGE` --
    serving it spent nothing."""

    document = json.loads(text)
    if kind == "structured":
        return StructuredResponse(
            data=document["data"],
            text=document["text"],
            usage=ZERO_USAGE,
            model=model,
            finish_reason=document.get("finish_reason"),
            native=bool(document.get("native", False)),
            attempts=int(document.get("attempts", 1)),
        )
    return LLMResponse(
        text=document["text"],
        usage=ZERO_USAGE,
        model=model,
        finish_reason=document.get("finish_reason"),
        tool_calls=tuple(ToolCall(**call) for call in document.get("tool_calls") or ()),
        metadata=dict(document.get("metadata") or {}),
    )


@dataclass(frozen=True)
class CacheEntry:
    """One cached completion. ``usage`` is what the original call cost, kept for
    reporting only -- a hit still reports :data:`ZERO_USAGE`."""

    text: str
    created_at: float
    usage: TokenUsage | None = None


@runtime_checkable
class CacheBackend(Protocol):
    """Storage for :class:`CachingLLM`. Implementations must be thread-safe.

    Expiry is decided by ``CachingLLM`` (which owns the clock and the TTL), not by the
    backend, so every backend gets identical TTL semantics for free.
    """

    def get(self, key: str) -> CacheEntry | None: ...

    def put(self, key: str, entry: CacheEntry) -> None: ...

    def delete(self, key: str) -> bool: ...

    def clear(self) -> None: ...


class InMemoryCacheBackend:
    """Process-local LRU, capped at ``max_entries`` (oldest-used evicted first)."""

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class SQLiteCacheBackend:
    """Persistent backend on stdlib ``sqlite3``, shareable between processes.

    Same connection discipline as ``tools/rag/cache.py``'s ``QueryAnswerCache``: a
    connection per call (``sqlite3.Connection`` is not safe to share across threads), WAL
    so readers do not block on a writer, a lazy schema so constructing a backend never
    touches the disk, and every ``sqlite3.Error`` degrades to a miss / no-op.

    The file holds prompts' responses in plain text; place it where prompt content is
    allowed to rest.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> bool:
        if self._schema_ready:
            return True
        with self._schema_lock:
            if self._schema_ready:
                return True
            try:
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
                with self._connection() as conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS llm_cache ("
                        "cache_key TEXT PRIMARY KEY, "
                        "response_text TEXT NOT NULL, "
                        "created_at REAL NOT NULL, "
                        "prompt_tokens INTEGER, "
                        "completion_tokens INTEGER"
                        ")"
                    )
            except (OSError, sqlite3.Error):
                logger.warning("llm_cache schema init failed; treating as empty", exc_info=True)
                return False
            self._schema_ready = True
            return True

    def get(self, key: str) -> CacheEntry | None:
        if not self._ensure_schema():
            return None
        try:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT response_text, created_at, prompt_tokens, completion_tokens "
                    "FROM llm_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
        except sqlite3.Error:
            logger.warning("llm_cache read failed; treating as a miss", exc_info=True)
            return None
        if row is None:
            return None
        text, created_at, prompt_tokens, completion_tokens = row
        usage = (
            None
            if prompt_tokens is None and completion_tokens is None
            else TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        )
        return CacheEntry(text=str(text), created_at=float(created_at), usage=usage)

    def put(self, key: str, entry: CacheEntry) -> None:
        if not self._ensure_schema():
            return
        usage = entry.usage
        try:
            with self._connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache "
                    "(cache_key, response_text, created_at, prompt_tokens, completion_tokens) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        key,
                        entry.text,
                        entry.created_at,
                        usage.prompt_tokens if usage else None,
                        usage.completion_tokens if usage else None,
                    ),
                )
        except sqlite3.Error:
            logger.warning("llm_cache write failed; response was not cached", exc_info=True)

    def delete(self, key: str) -> bool:
        if not self._ensure_schema():
            return False
        try:
            with self._connection() as conn:
                cursor = conn.execute("DELETE FROM llm_cache WHERE cache_key = ?", (key,))
                return cursor.rowcount > 0
        except sqlite3.Error:
            logger.warning("llm_cache delete failed", exc_info=True)
            return False

    def clear(self) -> None:
        if not self._ensure_schema():
            return
        try:
            with self._connection() as conn:
                conn.execute("DELETE FROM llm_cache")
        except sqlite3.Error:
            logger.warning("llm_cache clear failed", exc_info=True)


@dataclass(frozen=True)
class CacheStats:
    """Hit/miss counts for one :class:`CachingLLM` since construction."""

    hits: int
    misses: int

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


Namespace = str | Callable[[], str | None] | None


class CachingLLM(LLMWrapper):
    """Wraps an ``LLMProvider``, serving repeated identical calls from a backend.

    An :class:`~kuhaku.core.llm.wrapper.LLMWrapper`: ``generate``, ``chat``,
    ``generate_structured`` and ``chat_with_tools`` (and their async forms) are cached,
    each under its own key ``kind``; streams pass through uncached. A hit returns the
    stored response with :data:`ZERO_USAGE` and ``model`` set to this wrapper's name.

    ``params`` overrides the introspected generation parameters (see
    :func:`generation_params`) -- pass it for a provider whose sampling knobs are not
    in the attributes the bundled providers use. ``namespace`` adds a hard partition to
    every key: a string, or a zero-argument callable evaluated per call (e.g. reading a
    tenant from a ``contextvars.ContextVar``). ``ttl_seconds=None`` means entries never
    expire. ``clock`` exists for tests.

    Concurrent misses on the same key each call the provider; the cache does not
    coalesce them. That costs a duplicate call, never a wrong answer.

    Compose *inside* ``TokenTrackingLLM`` (``TokenTrackingLLM(CachingLLM(provider))``):
    this class reports :data:`ZERO_USAGE` as ``last_usage`` on a hit and the wrapped
    provider's usage on a miss, so tracked token counts reflect what was actually spent.
    """

    def __init__(
        self,
        wrapped: LLMProvider,
        backend: CacheBackend | None = None,
        *,
        ttl_seconds: float | None = None,
        params: Mapping[str, object] | None = None,
        namespace: Namespace = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive, or None for no expiry")
        super().__init__(wrapped)
        self._backend: CacheBackend = backend if backend is not None else InMemoryCacheBackend()
        self._ttl_seconds = ttl_seconds
        self._params = dict(params) if params is not None else generation_params(wrapped)
        self._namespace = namespace
        self._clock = clock
        self._counts_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        # last_usage is read by TokenTrackingLLM right after generate() returns, on the
        # same thread -- thread-local so concurrent calls cannot see each other's usage.
        self._local = threading.local()

    @property
    def backend(self) -> CacheBackend:
        return self._backend

    @property
    def last_usage(self) -> TokenUsage | None:
        return getattr(self._local, "usage", None)

    @property
    def stats(self) -> CacheStats:
        with self._counts_lock:
            return CacheStats(hits=self._hits, misses=self._misses)

    def cache_key(self, system: str, user: str) -> str:
        """The key ``generate(system, user)`` would use right now."""

        return self._key("generate", {"system": system, "user": user})

    def _key(self, kind: str, request: Mapping[str, object]) -> str:
        namespace = self._namespace() if callable(self._namespace) else self._namespace
        return llm_cache_key(
            self._wrapped.name, request, kind=kind, params=self._params, namespace=namespace
        )

    def generate(self, system: str, user: str) -> str:
        key, hit = self._generate_hit(system, user)
        if hit is not None:
            return hit.text
        response = generate_response(self._wrapped, system, user)
        self._generate_store(key, response)
        return response.text

    # ``generate(s, u)`` and ``chat([system(s), user(u)])`` are one logical call (see
    # ``wrapper.generate_call``): both are keyed and stored as ``kind="generate"``, so a
    # TokenTrackingLLM that routes agenerate through achat, or a caller mixing sync and
    # async, hits what the other form stored instead of paying for it twice.

    def _generate_hit(self, system: str, user: str) -> tuple[str, LLMResponse | None]:
        key = self.cache_key(system, user)
        entry = self._lookup(key)
        observe_lookup(source="cache", hit=entry is not None, provider_name=self.name)
        self._count(hit=entry is not None)
        if entry is None:
            return key, None
        self._local.usage = ZERO_USAGE
        return key, LLMResponse(text=entry.text, usage=ZERO_USAGE, model=self.name)

    def _generate_store(self, key: str, response: LLMResponse) -> None:
        self._local.usage = response.usage
        try:
            self._backend.put(
                key,
                CacheEntry(text=response.text, created_at=self._clock(), usage=response.usage),
            )
        except Exception:
            logger.exception("llm cache store failed (answer unaffected)")

    # --- v2 ---------------------------------------------------------------------------

    def chat(
        self, messages: Sequence[Message], *, options: GenerationOptions | None = None
    ) -> LLMResponse:
        call = generate_call(messages, options)
        if call is not None:
            key, hit = self._generate_hit(*call)
            if hit is not None:
                return hit
            response = super().chat(messages, options=options)
            self._generate_store(key, response)
            return response
        return self._serve(
            "chat",
            chat_request(messages, options),
            lambda: super(CachingLLM, self).chat(messages, options=options),
        )

    def generate_structured(
        self,
        messages: Sequence[Message],
        json_schema: dict[str, Any],
        *,
        name: str = "response",
        options: GenerationOptions | None = None,
        max_repair_attempts: int = 1,
    ) -> StructuredResponse:
        return self._serve(
            "structured",
            structured_request(messages, json_schema, name, options, max_repair_attempts),
            lambda: super(CachingLLM, self).generate_structured(
                messages,
                json_schema,
                name=name,
                options=options,
                max_repair_attempts=max_repair_attempts,
            ),
        )

    def chat_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        tool_choice: str = "auto",
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        return self._serve(
            "tools",
            tools_request(messages, tools, tool_choice, options),
            lambda: super(CachingLLM, self).chat_with_tools(
                messages, tools, tool_choice=tool_choice, options=options
            ),
        )

    def _hit(self, kind: str, request: Mapping[str, object]) -> tuple[str, Any]:
        key = self._key(kind, request)
        entry = self._lookup(key)
        response = None
        if entry is not None:
            try:
                response = decode_response(kind, entry.text, self.name)
            except (ValueError, KeyError, TypeError):
                logger.warning("unreadable llm cache entry; treating as a miss", exc_info=True)
        observe_lookup(source="cache", hit=response is not None, provider_name=self.name)
        self._count(hit=response is not None)
        if response is not None:
            self._local.usage = ZERO_USAGE
        return key, response

    def _store(self, key: str, response: LLMResponse | StructuredResponse) -> None:
        self._local.usage = response.usage
        text = encode_response(response)
        if text is None:
            return
        try:
            self._backend.put(
                key, CacheEntry(text=text, created_at=self._clock(), usage=response.usage)
            )
        except Exception:
            logger.exception("llm cache store failed (answer unaffected)")

    def _serve(
        self, kind: str, request: Mapping[str, object], call: Callable[[], V2Response]
    ) -> V2Response:
        key, hit = self._hit(kind, request)
        if hit is not None:
            return hit  # type: ignore[no-any-return]
        response = call()
        self._store(key, response)
        return response

    def invalidate(self, system: str, user: str) -> bool:
        """Drop the entry ``generate(system, user)`` would hit; True if one existed."""

        return self._backend.delete(self.cache_key(system, user))

    def clear(self) -> None:
        """Drop every entry in the backend -- including other wrappers' entries if the
        backend is shared."""

        self._backend.clear()

    def _lookup(self, key: str) -> CacheEntry | None:
        try:
            entry = self._backend.get(key)
        except Exception:
            logger.exception("llm cache lookup failed; treating as a miss")
            return None
        if entry is None:
            return None
        if self._ttl_seconds is not None and self._clock() - entry.created_at > self._ttl_seconds:
            try:
                self._backend.delete(key)
            except Exception:
                logger.debug("expired llm cache entry could not be removed", exc_info=True)
            return None
        return entry

    def _count(self, *, hit: bool) -> None:
        with self._counts_lock:
            if hit:
                self._hits += 1
            else:
                self._misses += 1
