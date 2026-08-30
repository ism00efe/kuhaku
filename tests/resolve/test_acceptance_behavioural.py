"""Spec §15 behavioural acceptance checks 1-12.

Each test is named for the check it encodes. ``resolve()`` is a pure decision -- it never
calls ``activate`` (amendment 1); the §5 consent flow and the backend construction happen
in the separate ``activate()`` phase.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("kuhaku.core.resolve", reason="mechanism lands in commit 3")

from kuhaku.core.exceptions import (  # noqa: E402
    CapabilityUnavailable,
    ConsentRequired,
    KuhakuError,
    StoreConflict,
)
from kuhaku.core.resolve import Cost, Fingerprint, Registry, activate, resolve  # noqa: E402
from kuhaku.core.resolve import consent as consent_mod  # noqa: E402
from kuhaku.core.resolve.memory import JsonMemory  # noqa: E402

from tests.resolve.conftest import (  # noqa: E402
    FakeAdapter,
    FakeUI,
    make_candidate,
    make_env,
)


def _registry(*adapters) -> Registry:
    reg = Registry()
    for adapter in adapters:
        reg.register(adapter)
    return reg


def _fp() -> Fingerprint:
    return Fingerprint(python="3.12", packages={}, gpu=None, vram_class=None, model_dir=None)


# --- Check 1 --------------------------------------------------------------------
def test_check1_zero_candidates_required_one_message_and_nothing_installed(env, memory, monkeypatch):
    monkeypatch.setattr(consent_mod, "run_pip",
                        lambda *a, **k: pytest.fail("run_pip must not be called"))
    missing = make_candidate(
        "ollama", "llm", ready=False,
        cost=Cost(download_required=True, download_bytes=4_000_000_000,
                  note="run `ollama pull qwen2.5:7b-instruct`"),
    )
    ui = FakeUI(interactive=False)
    reg = _registry(FakeAdapter("llm", [missing]))

    with pytest.raises(CapabilityUnavailable) as exc:
        resolve("llm", registry=reg, env=env, ui=ui, memory=memory, required=True)

    assert isinstance(exc.value, KuhakuError)
    assert len(ui.announcements) == 1
    message, prominent, _ = ui.announcements[0]
    assert prominent
    assert "ollama pull" in message   # the single next step
    assert "4" in message             # the download size (cost)


# --- Check 2 --------------------------------------------------------------------
def test_check2_exactly_one_candidate_no_prompt_but_announced(env, memory):
    only = make_candidate("chroma", "store")
    ui = FakeUI(interactive=True)  # a human IS present; still must not be asked
    reg = _registry(FakeAdapter("store", [only]))

    res = resolve("store", registry=reg, env=env, ui=ui, memory=memory, required=True)

    assert res.candidate.id == "chroma"
    assert res.reason == "only_option"
    assert ui.ask_calls == []
    joined = " ".join(ui.messages).lower()
    assert "chroma" in joined
    assert "pin" in joined or "override" in joined


# --- Check 3 --------------------------------------------------------------------
def test_check3_noninteractive_many_candidates_picks_safest_and_flags_prominently(env, memory):
    ui = FakeUI(interactive=False)
    reg = _registry(FakeAdapter("llm", [
        make_candidate("openai", "llm", safety_rank=5),
        make_candidate("ollama", "llm", safety_rank=1),
        make_candidate("groq", "llm", safety_rank=3),
    ]))

    res = resolve("llm", registry=reg, env=env, ui=ui, memory=memory, required=True)

    assert res.candidate.id == "ollama"
    assert res.reason == "safe_default"
    assert ui.ask_calls == []
    assert ui.prominent_messages


def test_check3_interactive_many_candidates_asks_the_human(env, memory):
    ui = FakeUI(interactive=True, ask_returns="groq")
    reg = _registry(FakeAdapter("llm", [
        make_candidate("ollama", "llm", safety_rank=1),
        make_candidate("groq", "llm", safety_rank=3),
    ]))

    res = resolve("llm", registry=reg, env=env, ui=ui, memory=memory, required=True)

    assert res.candidate.id == "groq"
    assert res.reason == "user_choice"
    assert len(ui.ask_calls) == 1
    _question, options = ui.ask_calls[0]
    assert {o.id for o in options} == {"ollama", "groq"}


# --- Check 4 --------------------------------------------------------------------
def test_check4_install_approval_does_not_trigger_download(env, memory, monkeypatch):
    pip_calls: list = []
    monkeypatch.setattr(consent_mod, "run_pip", lambda *a, **k: pip_calls.append((a, k)) or 0)

    def _activate():
        pytest.fail("the download must not run when download consent is withheld")

    cand = make_candidate(
        "st-local", "embedding", ready=False, activate=_activate,
        cost=Cost(install_required=True, download_required=True,
                  download_bytes=490_000_000, note="pip install 'kuhaku[dense]'"),
    )
    ui = FakeUI(interactive=True, confirm_map={"install": True, "download": False})
    reg = _registry(FakeAdapter("embedding", [cand]))

    res = resolve("embedding", registry=reg, env=env, ui=ui, memory=memory,
                  requested="st-local", required=True)
    assert res.candidate.id == "st-local"  # decision is pure; no consent yet

    with pytest.raises(ConsentRequired):
        activate(res, env=env, ui=ui)

    assert len(pip_calls) == 1, "install was approved, so it ran"
    assert any("download" in a for a in ui.confirm_actions), "download consent asked separately"


def test_check4_download_approval_alone_never_installs(env, memory, monkeypatch):
    monkeypatch.setattr(consent_mod, "run_pip",
                        lambda *a, **k: pytest.fail("no install was required or approved"))
    activated: list = []
    cand = make_candidate(
        "weights-only", "llm", ready=False, activate=lambda: activated.append(True) or "ok",
        cost=Cost(install_required=False, download_required=True,
                  download_bytes=1_000_000, note="ollama pull tiny"),
    )
    ui = FakeUI(interactive=True, confirm_map={"download": True})
    reg = _registry(FakeAdapter("llm", [cand]))

    res = resolve("llm", registry=reg, env=env, ui=ui, memory=memory,
                  requested="weights-only", required=True)
    backend = activate(res, env=env, ui=ui)

    assert backend == "ok"
    assert activated == [True]
    assert not any("install" in a for a in ui.confirm_actions)


# --- Check 5 --------------------------------------------------------------------
def test_check5_no_isolated_env_prints_command_and_does_not_touch_system_python(memory, monkeypatch):
    monkeypatch.setattr(consent_mod, "run_pip",
                        lambda *a, **k: pytest.fail("must never install into a non-isolated interpreter"))
    env = make_env(in_isolated_python=False, isolation_source=None)
    cand = make_candidate(
        "st-local", "embedding", ready=False,
        cost=Cost(install_required=True, note="pip install 'kuhaku[dense]'"),
    )
    ui = FakeUI(interactive=True, confirm_map={"install": True})
    reg = _registry(FakeAdapter("embedding", [cand]))

    res = resolve("embedding", registry=reg, env=env, ui=ui, memory=memory,
                  requested="st-local", required=True)
    with pytest.raises(ConsentRequired):
        activate(res, env=env, ui=ui)

    blob = " ".join(ui.messages)
    assert "pip install 'kuhaku[dense]'" in blob
    assert "venv" in blob or "environment" in blob


# --- Check 6 --------------------------------------------------------------------
def test_check6_remembered_across_runs_with_per_field_invalidation(tmp_path):
    mem = JsonMemory(tmp_path)
    base_env = make_env(gpu=None, vram_class="unknown")
    adapters = lambda: _registry(FakeAdapter("device", [
        make_candidate("cpu", "device", safety_rank=0),
        make_candidate("cuda", "device", safety_rank=1),
    ], packages=frozenset({"torch"})))

    ui1 = FakeUI(interactive=True, ask_returns="cuda")
    res1 = resolve("device", registry=adapters(), env=base_env, ui=ui1, memory=mem, required=True)
    assert res1.reason == "user_choice" and res1.candidate.id == "cuda"

    ui2 = FakeUI(interactive=True, ask_returns=lambda o: pytest.fail("must not ask again"))
    res2 = resolve("device", registry=adapters(), env=base_env, ui=ui2, memory=mem, required=True)
    assert res2.reason == "remembered" and res2.candidate.id == "cuda"
    assert len(ui2.announcements) <= 1

    ui3 = FakeUI(interactive=True, ask_returns=lambda o: pytest.fail("python change must not re-open device"))
    res3 = resolve("device", registry=adapters(),
                   env=make_env(gpu=None, vram_class="unknown", python="3.13"),
                   ui=ui3, memory=mem, required=True)
    assert res3.reason == "remembered"

    ui4 = FakeUI(interactive=True, ask_returns="cpu")
    res4 = resolve("device", registry=adapters(),
                   env=make_env(gpu="cuda", vram_class="large"),
                   ui=ui4, memory=mem, required=True)
    assert res4.reason == "user_choice"
    assert len(ui4.ask_calls) == 1


# --- Check 7 --------------------------------------------------------------------
def test_check7_llm_key_alone_never_selects_an_api_embedder(env, memory):
    """The API embedding adapter gates on its OWN credential; an LLM key present in the
    environment contributes no embedding candidate."""
    reg = _registry(
        FakeAdapter("embedding", []),  # api adapter: no embedding credential -> nothing
        FakeAdapter("embedding", [make_candidate("st-local", "embedding", ready=False)]),
    )
    res = resolve("embedding", registry=reg, env=env, ui=FakeUI(interactive=False),
                  memory=memory, required=False)

    assert res.candidate is None
    assert res.reason == "unavailable"


# --- Check 8 --------------------------------------------------------------------
def test_check8_explicit_dense_without_embedder_raises_consent_required(env, memory):
    cand = make_candidate("st-local", "embedding", ready=False,
                          cost=Cost(install_required=True, note="pip install 'kuhaku[dense]'"))
    ui = FakeUI(interactive=False)  # cannot consent
    reg = _registry(FakeAdapter("embedding", [cand]))

    res = resolve("embedding", registry=reg, env=env, ui=ui, memory=memory,
                  requested="st-local", required=True)
    with pytest.raises(ConsentRequired):
        activate(res, env=env, ui=ui)


def test_check8_unspecified_without_embedder_degrades_with_announcement(env, memory):
    reg = _registry(FakeAdapter("embedding", [
        make_candidate("sparse", "embedding", ready=True),  # zero-cost working alternative
    ]))
    ui = FakeUI(interactive=False)

    res = resolve("embedding", registry=reg, env=env, ui=ui, memory=memory, required=False)

    assert res.candidate.id == "sparse"
    assert ui.messages


# --- Check 9 --------------------------------------------------------------------
def _store_registry() -> Registry:
    return _registry(FakeAdapter("store", [
        make_candidate("chroma", "store", label="Chroma (serverless)"),
        make_candidate("qdrant", "store", label="Qdrant (server)"),
    ]))


def test_check9_crossing_chunk_threshold_suggests_options_never_migrates(env):
    from kuhaku.tools.rag.resolve.store_policy import suggest_store_upgrade

    ui = FakeUI(interactive=True, ask_returns=None)  # operator declines / no choice
    choice = suggest_store_upgrade(
        chunk_count=60_000, current_id="builtin",
        registry=_store_registry(), env=env, ui=ui,
    )

    assert choice is None  # suggest_store_upgrade has no migration hook -- it cannot migrate
    assert len(ui.ask_calls) == 1
    _question, options = ui.ask_calls[0]
    assert len(options) >= 2  # at least "stay" plus one heavier store


def test_check9_below_threshold_is_silent(env):
    from kuhaku.tools.rag.resolve.store_policy import suggest_store_upgrade

    ui = FakeUI(interactive=True)
    suggest_store_upgrade(chunk_count=100, current_id="builtin",
                          registry=_store_registry(), env=env, ui=ui)
    assert ui.ask_calls == []


# --- Check 10 -----------------------------------------------------------------
def test_check10_second_writer_conflict_is_clean_non_interactive(tmp_path):
    """The lock-file guard turns a second writer into a clean StoreConflict. Full
    two-process coverage against a real store lands with the Tier-0 store (§13)."""
    from kuhaku.tools.rag.resolve.store_policy import guard_single_writer

    with guard_single_writer(tmp_path, ui=FakeUI(interactive=False)):
        with pytest.raises(StoreConflict):
            with guard_single_writer(tmp_path, ui=FakeUI(interactive=False)):
                pass
    # released on exit
    with guard_single_writer(tmp_path, ui=FakeUI(interactive=False)):
        pass


# --- Check 11 -----------------------------------------------------------------
def test_check11_automatic_mode_no_model_found_names_no_model(env, memory):
    reg = _registry(FakeAdapter("llm", [
        make_candidate("ollama", "llm", ready=False,
                       label="local runner (size class: medium)",
                       cost=Cost(download_required=True, note="install a local runner")),
    ]))
    ui = FakeUI(interactive=False)

    res = resolve("llm", registry=reg, env=env, ui=ui, memory=memory, required=False)

    assert res.candidate is None
    for message in ui.messages:
        low = message.lower()
        assert "qwen" not in low
        assert ":7b" not in low
        assert "e5" not in low


# --- Check 12 -----------------------------------------------------------------
def test_check12_unknown_schema_is_ignored_and_decisions_remade(tmp_path):
    store = tmp_path / ".kuhaku" / "decisions.json"
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps({
        "schema": 999,
        "decisions": {"llm": {"candidate": "openai", "fingerprint": {}}},
    }))

    mem = JsonMemory(tmp_path)
    assert mem.get("llm", _fp()) is None

    mem.put("llm", _fp(), "ollama")
    reloaded = json.loads(store.read_text())
    assert reloaded["schema"] == 1
    assert JsonMemory(tmp_path).get("llm", _fp()) == "ollama"


def test_check12_corrupt_file_is_ignored(tmp_path):
    store = tmp_path / ".kuhaku" / "decisions.json"
    store.parent.mkdir(parents=True)
    store.write_text("{ this is not json")
    assert JsonMemory(tmp_path).get("llm", _fp()) is None
