"""Behaviours added in response to PR #8 review findings."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("kuhaku.core.resolve", reason="mechanism lands in commit 3")

from kuhaku.core.resolve import Fingerprint, Registry, probes, resolve  # noqa: E402
from kuhaku.core.resolve.memory import JsonMemory  # noqa: E402

from tests.resolve.conftest import FakeAdapter, FakeMemory, FakeUI, make_candidate, make_env


def _registry(*adapters) -> Registry:
    reg = Registry()
    for a in adapters:
        reg.register(a)
    return reg


def _fp() -> Fingerprint:
    return Fingerprint(python="3.12", packages={}, gpu=None, vram_class=None, model_dir=None)


# --- finding 2: a remembered choice that is no longer usable is announced -----
def test_stale_remembered_choice_is_announced_before_redeciding():
    reg = _registry(FakeAdapter("llm", [
        make_candidate("openai", "llm", ready=False),   # the remembered pick, credential gone
        make_candidate("ollama", "llm", ready=True, safety_rank=0),
    ]))
    mem = FakeMemory()
    mem.put("llm", _fp(), "openai")
    ui = FakeUI(interactive=False)

    res = resolve("llm", registry=reg, env=make_env(), ui=ui, memory=mem, required=True)

    assert res.candidate.id == "ollama"
    assert res.reason == "only_option"
    assert any("no longer usable" in m and "openai" in m for m in ui.messages)


def test_remembered_choice_still_usable_is_not_flagged_as_stale():
    reg = _registry(FakeAdapter("llm", [make_candidate("openai", "llm", ready=True)]))
    mem = FakeMemory()
    mem.put("llm", _fp(), "openai")
    ui = FakeUI(interactive=False)

    res = resolve("llm", registry=reg, env=make_env(), ui=ui, memory=mem, required=True)

    assert res.reason == "remembered"
    assert not any("no longer usable" in m for m in ui.messages)


# --- finding 3: schema mismatch is announced, symmetric with the unwritable path ---
def test_schema_mismatch_is_announced_once(tmp_path):
    store = tmp_path / ".kuhaku" / "decisions.json"
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps({"schema": 99, "decisions": {"llm": {"candidate": "x"}}}))

    ui = FakeUI(interactive=False)
    mem = JsonMemory(tmp_path, ui=ui)
    assert mem.get("llm", _fp()) is None
    assert mem.get("store", _fp()) is None  # second read: still no crash, no second message

    schema_msgs = [m for m in ui.messages if "schema" in m.lower()]
    assert len(schema_msgs) == 1


def test_schema_mismatch_without_ui_is_silent(tmp_path):
    store = tmp_path / ".kuhaku" / "decisions.json"
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps({"schema": 99, "decisions": {}}))
    assert JsonMemory(tmp_path).get("llm", _fp()) is None  # no ui -> no announce, no error


# --- finding 4: hf_cache_roots covers the env vars real deployments set ------
def test_hf_cache_roots_includes_configured_locations(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    st = tmp_path / "st"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st))
    roots = probes.hf_cache_roots()
    assert hub in roots
    assert st in roots
    # the default location is always a fallback
    assert any(r.name == "hub" and "huggingface" in str(r) for r in roots)


def test_model_cached_needs_an_actual_snapshot(monkeypatch, tmp_path):
    from kuhaku.tools.rag.resolve.adapters import embedding as emb

    monkeypatch.setattr(emb, "hf_cache_roots", lambda: [tmp_path])
    monkeypatch.setattr(emb.Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    model_dir = tmp_path / "models--intfloat--multilingual-e5-small"
    (model_dir / "snapshots").mkdir(parents=True)
    assert emb._model_cached("intfloat/multilingual-e5-small") is False  # snapshots dir empty

    (model_dir / "snapshots" / "abc123").mkdir()
    assert emb._model_cached("intfloat/multilingual-e5-small") is True


# --- second-review BLOCKER: a failed rename leaves no .tmp behind --------------
def test_write_failure_does_not_leak_a_tmp_file(tmp_path, monkeypatch):
    mem = JsonMemory(tmp_path)

    def _boom(self, target):
        raise OSError("rename failed")

    monkeypatch.setattr("pathlib.Path.replace", _boom)
    mem.put("llm", _fp(), "ollama")  # must not raise

    assert list((tmp_path / ".kuhaku").glob("*.tmp")) == []


# --- second-review: a raising adapter is logged, not silently swallowed -------
def test_adapter_that_raises_in_probe_is_logged(caplog):
    class _BrokenAdapter:
        kind = "llm"
        packages = frozenset()

        def probe(self, env):
            raise RuntimeError("adapter bug")

        def baseline(self, env):
            return None

    reg = _registry(_BrokenAdapter(), FakeAdapter("llm", [make_candidate("ollama", "llm")]))
    with caplog.at_level("WARNING", logger="kuhaku"):
        cands = reg.candidates("llm", make_env())
    assert [c.id for c in cands] == ["ollama"]  # the good adapter still contributes
    assert any("adapter bug" in r.message or "_BrokenAdapter" in r.message for r in caplog.records)


# --- second-review: KUHAKU_AUTO=false refuses a model download ----------------
def test_auto_disabled_refuses_to_download_the_embedding_model(monkeypatch, tmp_path):
    import kuhaku
    from kuhaku import RAG
    from kuhaku.core.config import Settings
    from kuhaku.core.exceptions import CapabilityUnavailable
    from tests.conftest import FakeLLM, FakeVectorStore

    monkeypatch.setenv("KUHAKU_AUTO", "false")
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: FakeVectorStore())
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s, **k: FakeLLM())
    monkeypatch.setattr(
        "kuhaku.tools.rag.resolve.adapters.embedding._model_cached", lambda name: False
    )

    def _must_not_build(rs, **k):
        raise AssertionError("KUHAKU_AUTO=false must not download the model")

    monkeypatch.setattr(kuhaku, "build_embedding_provider", _must_not_build)

    with pytest.raises(CapabilityUnavailable):
        RAG(settings=Settings(_env_file=None, audit_enabled=False), cache=False)
