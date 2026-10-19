"""Only an explicit user choice is remembered: an "only option" (or a safe default) is
re-derived every run, so a safer candidate that appears later wins."""

from __future__ import annotations

import json

from kuhaku.core.resolve import Registry, resolve
from kuhaku.core.resolve.memory import JsonMemory
from tests.resolve.conftest import FakeAdapter, FakeUI, make_candidate, make_env


def _registry(*candidates) -> Registry:
    reg = Registry()
    reg.register(FakeAdapter("llm", list(candidates)))
    return reg


def test_only_option_cloud_does_not_stick_after_local_appears(tmp_path):
    mem = JsonMemory(tmp_path)
    env = make_env()
    cloud = make_candidate("openai", "llm", safety_rank=5)
    first = resolve("llm", registry=_registry(cloud), env=env, ui=FakeUI(), memory=mem, required=True)
    assert first.reason == "only_option"
    assert not (tmp_path / ".kuhaku" / "decisions.json").exists()

    local = make_candidate("ollama", "llm", safety_rank=0)
    second = resolve(
        "llm", registry=_registry(cloud, local), env=env, ui=FakeUI(), memory=mem, required=True
    )
    assert second.candidate.id == "ollama"
    assert second.reason == "safe_default"


def test_safe_default_is_not_remembered(tmp_path):
    mem = JsonMemory(tmp_path)
    reg = _registry(make_candidate("a", "llm", safety_rank=1), make_candidate("b", "llm", safety_rank=2))
    resolve("llm", registry=reg, env=make_env(), ui=FakeUI(), memory=mem, required=True)
    assert not (tmp_path / ".kuhaku" / "decisions.json").exists()


def test_legacy_entry_without_reason_is_ignored(tmp_path):
    path = tmp_path / ".kuhaku" / "decisions.json"
    path.parent.mkdir()
    mem = JsonMemory(tmp_path)
    env = make_env()
    reg = _registry(make_candidate("openai", "llm", safety_rank=5), make_candidate("ollama", "llm", safety_rank=0))
    # Write a legacy entry through the real fingerprint, then strip its reason.
    resolve("llm", registry=reg, env=env, ui=FakeUI(interactive=True, ask_returns="openai"),
            memory=mem, required=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["decisions"]["llm"]["reason"] == "user_choice"
    del data["decisions"]["llm"]["reason"]
    path.write_text(json.dumps(data), encoding="utf-8")

    again = resolve("llm", registry=reg, env=env, ui=FakeUI(), memory=mem, required=True)
    assert again.reason == "safe_default" and again.candidate.id == "ollama"
