"""The generic probes and the KUHAKU_AUTO switch (moved here from the old
kuhaku.core.capabilities)."""

from __future__ import annotations

import pytest

pytest.importorskip("kuhaku.core.resolve", reason="mechanism lands in commit 3")

from kuhaku.core.resolve import auto_enabled  # noqa: E402
from kuhaku.core.resolve import probes  # noqa: E402


def test_auto_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("KUHAKU_AUTO", raising=False)
    assert auto_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
def test_auto_enabled_false_values(monkeypatch, value):
    monkeypatch.setenv("KUHAKU_AUTO", value)
    assert auto_enabled() is False


def test_module_available():
    assert probes.module_available("json") is True
    assert probes.module_available("nonexistent_module_xyz") is False


def test_endpoint_reachable_false_for_dead_port():
    assert probes.endpoint_reachable("http://127.0.0.1:9", timeout=0.2) is False


def test_loopback_daemon_reachable_rejects_non_loopback_host():
    # example.com is not loopback -> False without any connection attempt
    assert probes.loopback_daemon_reachable("http://example.com:11434") is False


def test_loopback_daemon_reachable_false_for_dead_loopback_port():
    assert probes.loopback_daemon_reachable("http://127.0.0.1:9") is False


def test_torch_accelerator_returns_a_known_value():
    assert probes.torch_accelerator() in ("cpu", "cuda", "mps")


def test_torch_accelerator_cpu_without_torch(monkeypatch):
    monkeypatch.setattr(probes, "module_available", lambda name: name != "torch")
    assert probes.torch_accelerator() == "cpu"


def test_detect_isolation_reports_venv(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/.venv")
    assert probes.detect_isolation() == (True, "venv")


def test_detect_isolation_no_env(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(probes.sys, "base_prefix", probes.sys.prefix, raising=False)
    isolated, source = probes.detect_isolation()
    assert isolated is False and source is None
