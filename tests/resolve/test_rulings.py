"""Behaviours fixed by the approved rulings and amendments that are not numbered
§15 checks."""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("kuhaku.core.resolve", reason="mechanism lands in commit 3")

from kuhaku.core.exceptions import (  # noqa: E402
    CapabilityUnavailable,
    CustomComponentError,
    FallbackWarning,
    KuhakuError,
    SecurityComponentError,
)
from kuhaku.core.resolve import Registry, activate, resolve  # noqa: E402
from kuhaku.core.resolve.memory import JsonMemory  # noqa: E402
from kuhaku.core.resolve.ui import ConsoleUI  # noqa: E402

from tests.resolve.conftest import (  # noqa: E402
    FakeAdapter,
    FakeMemory,
    FakeUI,
    make_candidate,
    make_env,
)


def _registry(*adapters) -> Registry:
    reg = Registry()
    for a in adapters:
        reg.register(a)
    return reg


# --- Ruling: error root ------------------------------------------------------
def test_existing_exceptions_reparented_under_kuhaku_error():
    assert issubclass(SecurityComponentError, KuhakuError)
    assert issubclass(CustomComponentError, KuhakuError)
    try:
        raise SecurityComponentError("x")
    except KuhakuError:
        pass


def test_fallback_warning_stays_outside_the_hierarchy():
    assert issubclass(FallbackWarning, Warning)
    assert not issubclass(FallbackWarning, KuhakuError)


# --- Amendment 1: resolve() is a pure decision -----------------------------
def test_resolve_never_calls_activate(env, memory):
    def _boom():
        pytest.fail("resolve() must not call Candidate.activate")

    reg = _registry(FakeAdapter("llm", [make_candidate("ollama", "llm", activate=_boom)]))
    res = resolve("llm", registry=reg, env=env, ui=FakeUI(), memory=memory, required=True)
    assert res.candidate.id == "ollama"


# --- Amendment 2: KUHAKU_AUTO=false short-circuits before any probing ------
def test_auto_disabled_returns_baseline_without_probing_or_memory(monkeypatch):
    monkeypatch.setenv("KUHAKU_AUTO", "false")
    adapter = FakeAdapter(
        "llm",
        [make_candidate("ollama", "llm"), make_candidate("groq", "llm")],
        baseline_id="ollama",
    )
    mem = FakeMemory()
    ui = FakeUI(interactive=True)
    res = resolve("llm", registry=_registry(adapter), env=make_env(), ui=ui,
                  memory=mem, required=True)

    assert res.candidate.id == "ollama"
    assert res.reason == "auto_disabled"
    assert adapter.probe_calls == 0          # no probing
    assert ui.ask_calls == []
    assert mem.get_calls == [] and mem.put_calls == []  # memory neither read nor written


def test_auto_disabled_announced_once_per_process(monkeypatch):
    monkeypatch.setenv("KUHAKU_AUTO", "false")
    adapter = FakeAdapter("llm", [make_candidate("ollama", "llm")], baseline_id="ollama")
    ui = FakeUI(interactive=False)
    for _ in range(3):
        resolve("llm", registry=_registry(adapter), env=make_env(), ui=ui,
                memory=FakeMemory(), required=True)
    assert sum("auto disabled" in m.lower() for m in ui.messages) == 1


def test_auto_disabled_missing_baseline_raises_naming_the_switch(monkeypatch):
    monkeypatch.setenv("KUHAKU_AUTO", "false")
    adapter = FakeAdapter("llm", [make_candidate("groq", "llm")], baseline_id=None)
    with pytest.raises(CapabilityUnavailable) as exc:
        resolve("llm", registry=_registry(adapter), env=make_env(), ui=FakeUI(),
                memory=FakeMemory(), required=True)
    assert "KUHAKU_AUTO" in str(exc.value)


def test_auto_disabled_unusable_baseline_surfaces_at_activation(monkeypatch):
    monkeypatch.setenv("KUHAKU_AUTO", "false")

    def _unusable():
        raise RuntimeError("ollama server not running")

    adapter = FakeAdapter(
        "llm", [make_candidate("ollama", "llm", activate=_unusable)], baseline_id="ollama",
    )
    res = resolve("llm", registry=_registry(adapter), env=make_env(), ui=FakeUI(),
                  memory=FakeMemory(), required=True)
    with pytest.raises(CapabilityUnavailable) as exc:
        activate(res, env=make_env(), ui=FakeUI())
    assert "ollama" in str(exc.value) and "KUHAKU_AUTO" in str(exc.value)


# --- Amendment 3: memory is a convenience, never a dependency -------------
def test_unwritable_project_dir_degrades_and_does_not_break_resolution(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    mem = JsonMemory(blocker)  # .kuhaku/ cannot be created under a file

    # both operations must be silent no-ops rather than raising
    mem.put("llm", _fp(), "ollama")
    assert mem.get("llm", _fp()) is None


def test_memory_stores_selections_not_consent(env, monkeypatch):
    """A remembered selection must not skip §5 -- consent runs again on the next run."""
    from kuhaku.core.resolve import Cost
    from kuhaku.core.resolve import consent as consent_mod

    monkeypatch.setattr(consent_mod, "run_pip", lambda *a, **k: 0)

    cand = make_candidate(
        "st-local", "embedding", ready=False, activate=lambda: "backend",
        cost=Cost(install_required=True, download_required=True, download_bytes=1,
                  note="pip install 'kuhaku[dense]'"),
    )
    mem = FakeMemory()
    mem.put("embedding", _fp(), "st-local")  # a prior run's selection

    ui = FakeUI(interactive=True, confirm_map={"install": True, "download": True})
    res = resolve("embedding", registry=_registry(FakeAdapter("embedding", [cand])),
                  env=env, ui=ui, memory=mem, requested="st-local", required=True)
    activate(res, env=env, ui=ui)
    assert ui.confirm_calls, "consent was requested despite the remembered selection"


# --- Amendment 6: one isolation implementation ---------------------------
def test_consent_reads_environment_and_adds_no_detection():
    import inspect

    from kuhaku.core.resolve import consent

    src = inspect.getsource(consent)
    for token in ("VIRTUAL_ENV", "CONDA_PREFIX", "base_prefix"):
        assert token not in src, f"{token} detection must live only in probes.py"


# --- Ruling 2: degraded is orthogonal to prominent, and still warns ------
def test_degraded_announcement_emits_fallback_warning():
    ui = ConsoleUI()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ui.announce("dense retrieval unavailable; using sparse", degraded=True)
    assert any(isinstance(w.message, FallbackWarning) for w in caught)


# --- Ruling 5: baseline metadata on Resolution --------------------------
def test_resolution_carries_baseline_when_it_differs(env, memory):
    adapter = FakeAdapter(
        "llm",
        [make_candidate("ollama", "llm", safety_rank=1),
         make_candidate("groq", "llm", safety_rank=0)],
        baseline_id="ollama",
    )
    res = resolve("llm", registry=_registry(adapter), env=env,
                  ui=FakeUI(interactive=False), memory=memory, required=True)
    assert res.candidate.id == "groq"
    assert res.baseline == "ollama"


# --- Ruling 6: dedupe state lives on the UI instance -------------------
def test_two_console_uis_do_not_share_dedupe_state():
    a, b = ConsoleUI(), ConsoleUI()
    key = ("llm", "ollama", "only_option")
    a.announce("once", dedupe_key=key)
    a.announce("once", dedupe_key=key)
    before = len(getattr(b, "_announced", set()))
    b.announce("once", dedupe_key=key)
    assert len(b._announced) == before + 1


# --- local helpers -----------------------------------------------------
def _fp():
    from kuhaku.core.resolve import Fingerprint

    return Fingerprint(python="3.12", packages={}, gpu=None, vram_class=None, model_dir=None)
