"""Tests for the CachingLLM decorator and its backends."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import replace

import pytest

from kuhaku.core.llm.base import LLMProvider, TokenUsage
from kuhaku.core.llm.cache import (
    ZERO_USAGE,
    CacheBackend,
    CacheEntry,
    CachingLLM,
    InMemoryCacheBackend,
    SQLiteCacheBackend,
    generation_params,
    llm_cache_key,
)
from kuhaku.core.llm.token_tracking import TokenTrackingLLM
from kuhaku.core.observability.metrics import LLM_CACHE_LOOKUPS, LLM_TOKENS_TOTAL
from kuhaku.core.observability.tracing import GEN_AI_SYSTEM, GEN_AI_TOKEN_TYPE
from tests.conftest import prometheus_counter_value as _counter_value


class _CountingProvider:
    """Answers ``"<n>:<user>"`` with an incrementing call number, so a cached response
    is distinguishable from a fresh one."""

    def __init__(self, name: str = "fake:model-x", usage: TokenUsage | None = None) -> None:
        self._name = name
        self.calls = 0
        self._usage = usage
        self.last_usage: TokenUsage | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    def generate(self, system: str, user: str) -> str:
        with self._lock:
            self.calls += 1
            if self._usage is not None:
                # A new object per call, as the real providers report usage.
                self.last_usage = replace(self._usage)
            return f"{self.calls}:{user}"


class _FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# --- key ---------------------------------------------------------------------------


def test_key_is_stable_and_order_independent():
    a = llm_cache_key("p:m", {"system": "s", "user": "u"}, params={"t": 0, "m": 5})
    b = llm_cache_key("p:m", {"user": "u", "system": "s"}, params={"m": 5, "t": 0})
    assert a == b
    assert len(a) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "p:other"},
        {"request": {"system": "s2", "user": "u"}},
        {"request": {"system": "s", "user": "u2"}},
        {"params": {"temperature": 0.7}},
        {"namespace": "tenant-a"},
        {"kind": "messages"},
    ],
)
def test_key_changes_with_every_output_affecting_field(change):
    base = {"provider": "p:m", "request": {"system": "s", "user": "u"}}
    assert llm_cache_key(**base) != llm_cache_key(**{**base, **change})


def test_key_fields_cannot_collide_by_concatenation():
    # "ab" + "c" vs "a" + "bc" must not hash the same.
    assert llm_cache_key("p:m", {"system": "ab", "user": "c"}) != llm_cache_key(
        "p:m", {"system": "a", "user": "bc"}
    )


def test_generation_params_reads_bundled_provider_attributes():
    class _P:
        _temperature = 0.2
        _max_tokens = 512
        _base_url = "http://x"
        _api_key = "secret"

    assert generation_params(_P()) == {"temperature": 0.2, "max_tokens": 512, "base_url": "http://x"}
    assert generation_params(object()) == {}


# --- decorator behaviour -------------------------------------------------------------


def test_satisfies_llm_provider_protocol():
    assert isinstance(CachingLLM(_CountingProvider()), LLMProvider)


def test_hit_serves_cached_text_without_calling_provider():
    provider = _CountingProvider()
    llm = CachingLLM(provider)
    assert llm.generate("sys", "q") == "1:q"
    assert llm.generate("sys", "q") == "1:q"
    assert provider.calls == 1
    assert llm.stats.hits == 1 and llm.stats.misses == 1
    assert llm.stats.hit_ratio == 0.5


def test_different_system_prompt_misses():
    provider = _CountingProvider()
    llm = CachingLLM(provider)
    llm.generate("sys-a", "q")
    assert llm.generate("sys-b", "q") == "2:q"


def test_name_delegates():
    assert CachingLLM(_CountingProvider(name="openai:gpt")).name == "openai:gpt"


def test_changed_generation_params_miss():
    backend = InMemoryCacheBackend()
    provider = _CountingProvider()
    CachingLLM(provider, backend, params={"temperature": 0}).generate("s", "q")
    CachingLLM(provider, backend, params={"temperature": 1}).generate("s", "q")
    assert provider.calls == 2


def test_namespace_partitions_a_shared_backend():
    backend = InMemoryCacheBackend()
    provider = _CountingProvider()
    a = CachingLLM(provider, backend, namespace="tenant-a")
    b = CachingLLM(provider, backend, namespace="tenant-b")
    a.generate("s", "q")
    assert b.generate("s", "q") == "2:q"
    assert a.generate("s", "q") == "1:q"


def test_callable_namespace_is_evaluated_per_call():
    current = {"tenant": "a"}
    provider = _CountingProvider()
    llm = CachingLLM(provider, namespace=lambda: current["tenant"])
    llm.generate("s", "q")
    current["tenant"] = "b"
    assert llm.generate("s", "q") == "2:q"
    current["tenant"] = "a"
    assert llm.generate("s", "q") == "1:q"


def test_ttl_expires_entries_with_injected_clock():
    clock = _FakeClock()
    provider = _CountingProvider()
    backend = InMemoryCacheBackend()
    llm = CachingLLM(provider, backend, ttl_seconds=60, clock=clock)
    llm.generate("s", "q")
    clock.now += 59
    assert llm.generate("s", "q") == "1:q"
    clock.now += 2
    assert llm.generate("s", "q") == "2:q"
    assert provider.calls == 2


def test_ttl_must_be_positive():
    with pytest.raises(ValueError):
        CachingLLM(_CountingProvider(), ttl_seconds=0)


def test_invalidate_and_clear():
    provider = _CountingProvider()
    llm = CachingLLM(provider)
    llm.generate("s", "q1")
    llm.generate("s", "q2")
    assert llm.invalidate("s", "q1") is True
    assert llm.invalidate("s", "q1") is False
    assert llm.generate("s", "q1") == "3:q1"
    llm.clear()
    assert llm.generate("s", "q2") == "4:q2"


def test_provider_errors_are_not_cached():
    class _Failing:
        name = "fake:x"
        calls = 0

        def generate(self, system, user):
            self.calls += 1
            raise RuntimeError("boom")

    provider = _Failing()
    llm = CachingLLM(provider)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            llm.generate("s", "q")
    assert provider.calls == 2


def test_broken_backend_degrades_to_passthrough():
    class _Broken:
        def get(self, key):
            raise OSError("disk gone")

        def put(self, key, entry):
            raise OSError("disk gone")

        def delete(self, key):
            raise OSError("disk gone")

        def clear(self):
            raise OSError("disk gone")

    provider = _CountingProvider()
    llm = CachingLLM(provider, _Broken())
    assert llm.generate("s", "q") == "1:q"
    assert llm.generate("s", "q") == "2:q"


# --- token usage ---------------------------------------------------------------------


def test_last_usage_is_zero_on_hit_and_passthrough_on_miss():
    usage = TokenUsage(prompt_tokens=10, completion_tokens=4)
    llm = CachingLLM(_CountingProvider(usage=usage))
    llm.generate("s", "q")
    assert llm.last_usage == usage
    llm.generate("s", "q")
    assert llm.last_usage == ZERO_USAGE


def test_token_tracking_records_zero_on_hit():
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
    tracked = TokenTrackingLLM(
        CachingLLM(_CountingProvider(name="cachetest:m", usage=usage)), provider="cachetest"
    )
    labels = {GEN_AI_SYSTEM: "cachetest", GEN_AI_TOKEN_TYPE: "input"}
    before = _counter_value(LLM_TOKENS_TOTAL, **labels)
    tracked.generate("s", "q")
    tracked.generate("s", "q")
    assert _counter_value(LLM_TOKENS_TOTAL, **labels) == before + 100


def test_last_usage_is_per_thread():
    llm = CachingLLM(_CountingProvider(usage=TokenUsage(1, 1)))
    llm.generate("s", "warm")  # main thread: a miss, real usage
    seen = []

    def worker():
        llm.generate("s", "warm")  # a hit in this thread
        seen.append(llm.last_usage)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen == [ZERO_USAGE]
    assert llm.last_usage == TokenUsage(1, 1)


# --- observability -------------------------------------------------------------------


def test_hit_and_miss_metrics():
    llm = CachingLLM(_CountingProvider(name="metricsprov:m"))
    labels = {"source": "cache", GEN_AI_SYSTEM: "metricsprov"}
    hits0 = _counter_value(LLM_CACHE_LOOKUPS, result="hit", **labels)
    miss0 = _counter_value(LLM_CACHE_LOOKUPS, result="miss", **labels)
    llm.generate("s", "q")
    llm.generate("s", "q")
    llm.generate("s", "q")
    assert _counter_value(LLM_CACHE_LOOKUPS, result="hit", **labels) == hits0 + 2
    assert _counter_value(LLM_CACHE_LOOKUPS, result="miss", **labels) == miss0 + 1


def test_span_attributes_set_on_lookup(monkeypatch):
    from kuhaku.core.llm import cache as cache_module

    seen: list[dict] = []
    monkeypatch.setattr(
        cache_module, "set_current_span_attributes", lambda **kw: seen.append(kw)
    )
    llm = CachingLLM(_CountingProvider())
    llm.generate("s", "q")
    llm.generate("s", "q")
    assert [a[cache_module.SPAN_CACHE_HIT] for a in seen] == [False, True]
    assert all(a[cache_module.SPAN_CACHE_SOURCE] == "cache" for a in seen)


def test_observability_failure_does_not_break_generate(monkeypatch):
    from kuhaku.core.llm import cache as cache_module

    def boom(**_):
        raise RuntimeError("metrics down")

    monkeypatch.setattr(cache_module, "record_llm_cache_lookup", boom)
    assert CachingLLM(_CountingProvider()).generate("s", "q") == "1:q"


# --- thread safety -------------------------------------------------------------------


def test_concurrent_generate_is_consistent():
    provider = _CountingProvider()
    llm = CachingLLM(provider, InMemoryCacheBackend(max_entries=64))
    prompts = [f"q{i % 8}" for i in range(400)]
    results: dict[str, set[str]] = {}
    lock = threading.Lock()

    def worker(chunk):
        for p in chunk:
            text = llm.generate("s", p)
            with lock:
                results.setdefault(p, set()).add(text)

    threads = [threading.Thread(target=worker, args=(prompts[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert set(results) == {f"q{i}" for i in range(8)}
    for p, texts in results.items():
        assert all(t.endswith(f":{p}") for t in texts)
    stats = llm.stats
    assert stats.hits + stats.misses == 400
    # Concurrent first misses may each call the provider, but never more than once per
    # thread per prompt.
    assert provider.calls == stats.misses <= 64


# --- backends ------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def backend(request, tmp_path) -> CacheBackend:
    if request.param == "memory":
        return InMemoryCacheBackend()
    return SQLiteCacheBackend(tmp_path / "nested" / "llm.sqlite3")


def test_backend_roundtrip(backend):
    assert isinstance(backend, CacheBackend)
    entry = CacheEntry(text="hello", created_at=5.0, usage=TokenUsage(3, 2))
    assert backend.get("k") is None
    backend.put("k", entry)
    assert backend.get("k") == entry
    backend.put("k", CacheEntry(text="new", created_at=6.0))
    assert backend.get("k") == CacheEntry(text="new", created_at=6.0, usage=None)
    assert backend.delete("k") is True
    assert backend.delete("k") is False
    backend.put("a", entry)
    backend.clear()
    assert backend.get("a") is None


def test_lru_evicts_least_recently_used():
    backend = InMemoryCacheBackend(max_entries=2)
    e = CacheEntry(text="x", created_at=0.0)
    backend.put("a", e)
    backend.put("b", e)
    backend.get("a")  # a is now most recent
    backend.put("c", e)
    assert backend.get("b") is None
    assert backend.get("a") is not None and backend.get("c") is not None
    assert len(backend) == 2


def test_lru_rejects_nonpositive_cap():
    with pytest.raises(ValueError):
        InMemoryCacheBackend(max_entries=0)


def test_sqlite_persists_across_instances(tmp_path):
    path = tmp_path / "llm.sqlite3"
    provider = _CountingProvider()
    CachingLLM(provider, SQLiteCacheBackend(path)).generate("s", "q")
    assert CachingLLM(provider, SQLiteCacheBackend(path)).generate("s", "q") == "1:q"
    assert provider.calls == 1


def test_sqlite_constructor_touches_no_disk(tmp_path):
    path = tmp_path / "sub" / "llm.sqlite3"
    SQLiteCacheBackend(path)
    assert not path.parent.exists()


def test_sqlite_errors_degrade_to_miss(tmp_path, monkeypatch):
    backend = SQLiteCacheBackend(tmp_path / "llm.sqlite3")
    backend.put("k", CacheEntry(text="x", created_at=0.0))

    def broken_connect(*a, **kw):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(sqlite3, "connect", broken_connect)
    assert backend.get("k") is None
    backend.put("k", CacheEntry(text="y", created_at=0.0))  # no raise
    assert backend.delete("k") is False


def test_sqlite_concurrent_writers(tmp_path):
    backend = SQLiteCacheBackend(tmp_path / "llm.sqlite3")
    errors: list[BaseException] = []

    def worker(n):
        try:
            for i in range(20):
                backend.put(f"{n}-{i}", CacheEntry(text=str(i), created_at=0.0))
                assert backend.get(f"{n}-{i}") is not None
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
