"""Unit tests for kuhaku.core.device.resolve_device."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from kuhaku.core.device import resolve_device


def _fake_torch(*, cuda: bool, mps: bool | None):
    """Build a minimal stand-in for the `torch` module -- `mps=None` simulates a torch
    build with no `torch.backends.mps` attribute at all (older torch versions)."""

    backends = SimpleNamespace()
    if mps is not None:
        backends.mps = SimpleNamespace(is_available=lambda: mps)
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=backends,
    )


def test_cpu_is_always_returned_as_is(monkeypatch):
    assert resolve_device("cpu", component="test") == "cpu"


def test_unknown_device_string_passed_through_unchanged():
    assert resolve_device("cuda:1", component="test") == "cuda:1"


def test_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False))
    assert resolve_device("auto", component="test") == "cuda"


def test_auto_prefers_mps_when_cuda_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
    assert resolve_device("auto", component="test") == "mps"


def test_auto_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
    assert resolve_device("auto", component="test") == "cpu"


def test_auto_falls_back_to_cpu_without_torch_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device("auto", component="test") == "cpu"


def test_explicit_cuda_falls_back_to_cpu_with_warning_when_unavailable(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
    with caplog.at_level("WARNING"):
        assert resolve_device("cuda", component="reranker") == "cpu"
    assert "reranker requested device 'cuda'" in caplog.text


def test_explicit_mps_falls_back_to_cpu_when_backend_missing(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=None))
    with caplog.at_level("WARNING"):
        assert resolve_device("mps", component="embedding model") == "cpu"
    assert "embedding model requested device 'mps'" in caplog.text


def test_explicit_cuda_used_when_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False))
    assert resolve_device("cuda", component="test") == "cuda"


@pytest.mark.parametrize("requested", ["AUTO", " Cuda ", "MPS"])
def test_device_names_are_case_and_whitespace_insensitive(monkeypatch, requested):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=True))
    assert resolve_device(requested, component="test") in ("cuda", "mps")
