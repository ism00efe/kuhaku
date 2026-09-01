"""Facade-level tests for the query-answer cache (FR2) -- kuhaku.RAG's `cache` kwarg.

Before this feature, `RAGSettings.cache_enabled`/`cache_ttl_seconds` had no reader in
src/, and `RAG.__init__` never built or passed a `QueryAnswerCache` to the engine --
`tests/rag/test_engine.py`'s cache tests (including the entitlement-isolation one)
construct `RAGEngine(..., cache=FakeCache())` directly, which never exercised that gap.
These tests go through `kuhaku.RAG` instead, with a real (SQLite-backed)
`QueryAnswerCache`.
"""

from __future__ import annotations

import logging

import kuhaku
from kuhaku import RAG
from kuhaku.core.auth import AuthContext
from kuhaku.core.config import Settings
from kuhaku.tools.rag.config import RAGSettings
from tests.conftest import FakeEmbeddings, FakeLLM, FakeVectorStore, make_chunk


def _patch_rag_deps(monkeypatch, fake_store, fake_embedder, fake_llm):
    monkeypatch.setattr(kuhaku, "build_embedding_provider", lambda rs: fake_embedder)
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: fake_store)
    monkeypatch.setattr(
        "kuhaku.tools.rag.resolve.adapters.embedding._model_cached", lambda name: True
    )
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s, **_k: fake_llm)


def _build_rag(monkeypatch, store, llm, *, rag_settings=None, **kwargs) -> RAG:
    _patch_rag_deps(monkeypatch, store, FakeEmbeddings(), llm)
    settings = Settings(_env_file=None, audit_enabled=False)
    return RAG(settings=settings, rag_settings=rag_settings, **kwargs)


# --- cache hit / TTL, through the facade --------------------------------------------
def test_facade_second_identical_question_is_served_from_the_cache(monkeypatch, tmp_path):
    store = FakeVectorStore([make_chunk("d1", text="alpha")])
    llm = FakeLLM(response="Answer [S1].")
    rag = _build_rag(monkeypatch, store, llm, cache=str(tmp_path / "cache.db"))

    first = rag.ask("soru")
    assert llm.last_user is not None  # miss -> LLM called

    llm.last_user = None
    second = rag.ask("soru")

    assert llm.last_user is None  # hit -> generate() never invoked
    assert second.text == first.text


def test_facade_cache_entry_expires_after_the_configured_ttl(monkeypatch, tmp_path):
    import kuhaku.tools.rag.cache as cache_module

    now = [1_000_000.0]
    monkeypatch.setattr(cache_module.time, "time", lambda: now[0])

    store = FakeVectorStore([make_chunk("d1", text="alpha")])
    llm = FakeLLM(response="Answer [S1].")
    rag_settings = RAGSettings(
        cache_enabled=True, cache_ttl_seconds=60, cache_db_path=str(tmp_path / "cache.db")
    )
    rag = _build_rag(monkeypatch, store, llm, rag_settings=rag_settings)

    rag.ask("soru")
    assert llm.last_user is not None
    llm.last_user = None

    now[0] += 30  # inside the 60s window
    rag.ask("soru")
    assert llm.last_user is None  # still a hit

    now[0] += 31  # 61s since the write -> past the TTL
    rag.ask("soru")
    assert llm.last_user is not None  # expired -> miss, LLM called again


# --- default behavior: cache_enabled=True means "configure nothing, still cached" ----
def test_facade_default_cache_is_enabled_at_the_managed_default_location(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = FakeVectorStore([make_chunk("d1", text="alpha")])
    llm = FakeLLM(response="Answer [S1].")
    rag = _build_rag(monkeypatch, store, llm)  # cache=None (default), rag_settings=None

    rag.ask("soru")
    llm.last_user = None
    rag.ask("soru")

    assert llm.last_user is None  # caching is on by default -- no kwarg configured it
    assert (tmp_path / "data" / "kuhaku_qa_cache.sqlite3").exists()


def test_facade_ragsettings_cache_enabled_false_disables_caching_by_default(monkeypatch, tmp_path):
    """Proves RAGSettings.cache_enabled is a real reader, not just a field: turning it
    off on RAGSettings (with the facade `cache` kwarg left at its None default) must
    disable caching, the same as passing `cache=False` explicitly."""

    store = FakeVectorStore([make_chunk("d1", text="alpha")])
    llm = FakeLLM(response="Answer [S1].")
    rag_settings = RAGSettings(cache_enabled=False, cache_db_path=str(tmp_path / "cache.db"))
    rag = _build_rag(monkeypatch, store, llm, rag_settings=rag_settings)

    rag.ask("soru")
    llm.last_user = None
    rag.ask("soru")

    assert llm.last_user is not None  # no cache -> LLM called every time
    assert not (tmp_path / "cache.db").exists()
    assert rag.engine.get_cache() is None


# --- cache=False: no file at all ------------------------------------------------------
def test_facade_cache_false_creates_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = FakeVectorStore([make_chunk("d1", text="alpha")])
    llm = FakeLLM(response="Answer [S1].")
    rag = _build_rag(monkeypatch, store, llm, cache=False)

    rag.ask("soru")

    assert rag.engine.get_cache() is None
    assert not (tmp_path / "data").exists()
    assert not any(tmp_path.rglob("*.sqlite3"))


# --- broken cache degrades gracefully -------------------------------------------------
def test_facade_broken_cache_directory_degrades_to_no_caching_with_warning(
    monkeypatch, tmp_path, caplog
):
    # A file where a directory needs to exist -- mkdir(parents=True) fails on every
    # platform, exercising the same degrade-with-warning path as rag/cache.py's own
    # sqlite3.Error catch, one layer earlier (directory creation, not connection).
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("x", encoding="utf-8")

    store = FakeVectorStore([make_chunk("d1", text="alpha")])
    llm = FakeLLM(response="Answer [S1].")
    rag = _build_rag(monkeypatch, store, llm, cache=str(blocked / "cache.db"))

    # RAG() construction itself never touches the cache's filesystem state (schema
    # creation is lazy -- see rag/cache.py) -- the failure, and its warning, surface on
    # first actual use, inside ask(). The query still succeeds; caching is simply inert.
    with caplog.at_level(logging.WARNING):
        ans = rag.ask("soru")

    assert ans.text == "Answer [S1]."
    assert "qa_cache" in caplog.text


# --- entitlement isolation, through the live cache and the facade --------------------
def test_facade_cache_does_not_leak_across_entitlements(monkeypatch, tmp_path):
    people_ops_chunk = make_chunk(
        "salary_policy", access_tags=("people_ops",), text="salary bands are A-E"
    )
    eng_chunk = make_chunk(
        "deploy_runbook", access_tags=("engineering",), text="deploy steps are 1-2-3"
    )
    store = FakeVectorStore([people_ops_chunk, eng_chunk])
    llm = FakeLLM(response="Answer [S1].")
    rag = _build_rag(monkeypatch, store, llm, cache=str(tmp_path / "cache.db"))

    hr_user = AuthContext(identity="hr1", roles=("people_ops",))
    eng_user = AuthContext(identity="eng1", roles=("engineering",))

    hr_answer = rag.ask("what information is available?", auth_context=hr_user)
    eng_answer = rag.ask("what information is available?", auth_context=eng_user)

    assert [rc.chunk.id for rc in hr_answer.retrieved] == [people_ops_chunk.id]
    assert [rc.chunk.id for rc in eng_answer.retrieved] == [eng_chunk.id]

    # Re-asking as eng must hit eng's own cache entry -- never hr's -- and stay grounded
    # in eng_chunk. A filter applied after the cache lookup would let hr's cached answer
    # leak straight into eng's response.
    llm.last_user = None
    eng_answer_again = rag.ask("what information is available?", auth_context=eng_user)

    assert llm.last_user is None  # served from eng's own cache entry -- still a hit
    assert eng_answer_again.text == eng_answer.text
    assert [rc.chunk.id for rc in eng_answer_again.retrieved] == [eng_chunk.id]


# --- two RAG instances sharing one cache file -----------------------------------------
def test_facade_two_rag_instances_share_a_cache_path_without_crashing(monkeypatch, tmp_path):
    cache_path = str(tmp_path / "shared_cache.db")
    chunk = make_chunk("d1", text="alpha")

    llm1 = FakeLLM(response="Answer [S1].")
    rag1 = _build_rag(monkeypatch, FakeVectorStore([chunk]), llm1, cache=cache_path)

    llm2 = FakeLLM(response="Answer [S1].")
    rag2 = _build_rag(monkeypatch, FakeVectorStore([chunk]), llm2, cache=cache_path)

    rag1.ask("soru")  # writes the shared cache entry

    rag2.ask("soru")  # second process/instance, same key -> hit against the shared file
    assert llm2.last_user is None
