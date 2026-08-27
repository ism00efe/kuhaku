"""Tests for kuhaku.tools.rag.capabilities: the RAG-specific ``"auto"`` chains.

The RAG counterpart to tests/test_capabilities.py, mirroring the source split
(kuhaku.core.capabilities vs kuhaku.tools.rag.capabilities) the same way
tests/rag/test_config.py mirrors tests/test_config.py.
"""

from __future__ import annotations

import warnings

import pytest

from kuhaku.core import capabilities as core_cap
from kuhaku.core.exceptions import FallbackWarning
from kuhaku.tools.rag import capabilities as rag_cap
from kuhaku.tools.rag.capabilities import (
    announce_retrieval_downgrade,
    resolve_embedding_device,
)


@pytest.fixture(autouse=True)
def _clear_emitted():
    core_cap.reset_emitted()
    yield
    core_cap.reset_emitted()


# --- resolve_embedding_device --------------------------------------------------------

@pytest.mark.parametrize("pinned", ["cpu", "cuda", "mps"])
def test_pinned_device_is_returned_untouched(pinned):
    assert resolve_embedding_device(pinned) == pinned


def test_auto_asks_torch_accelerator(monkeypatch):
    monkeypatch.setattr(rag_cap, "torch_accelerator", lambda: "cuda")
    assert resolve_embedding_device("auto") == "cuda"


def test_auto_none_is_treated_as_auto(monkeypatch):
    monkeypatch.setattr(rag_cap, "torch_accelerator", lambda: "mps")
    assert resolve_embedding_device(None) == "mps"


def test_auto_disabled_forces_cpu(monkeypatch):
    monkeypatch.setenv("KUHAKU_AUTO", "false")
    monkeypatch.setattr(rag_cap, "torch_accelerator", lambda: "cuda")
    assert resolve_embedding_device("auto") == "cpu"


def test_auto_cpu_is_silent(monkeypatch, capsys):
    monkeypatch.setattr(rag_cap, "torch_accelerator", lambda: "cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_embedding_device("auto") == "cpu"
    assert not caught
    assert capsys.readouterr().err == ""


def test_auto_accelerator_is_announced(monkeypatch, capsys):
    monkeypatch.setattr(rag_cap, "torch_accelerator", lambda: "cuda")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_embedding_device("auto")
    assert any(isinstance(w.message, FallbackWarning) for w in caught)
    assert "embedding_device" in capsys.readouterr().err


# --- torch_accelerator (the underlying core probe) -----------------------------------

def test_torch_accelerator_returns_cpu_without_torch(monkeypatch):
    monkeypatch.setattr(core_cap, "module_available", lambda name: name != "torch")
    assert core_cap.torch_accelerator() == "cpu"


def test_torch_accelerator_returns_a_known_value():
    assert core_cap.torch_accelerator() in ("cpu", "cuda", "mps")


# --- announce_retrieval_downgrade ---------------------------------------------------

def test_announce_retrieval_downgrade_emits_once(capsys):
    announce_retrieval_downgrade("torch missing")
    announce_retrieval_downgrade("torch missing")
    err = capsys.readouterr().err
    assert err.count("[kuhaku]") == 1
    assert "retrieval" in err and "sparse" in err
