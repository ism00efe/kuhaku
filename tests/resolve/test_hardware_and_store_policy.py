"""§12 constants: the VRAM size-class helper and the store-upgrade suggestion."""

from __future__ import annotations

import pytest

pytest.importorskip("kuhaku.core.resolve", reason="mechanism lands in commit 3")

from kuhaku.core.exceptions import StoreConflict  # noqa: E402
from kuhaku.core.resolve import Registry  # noqa: E402
from kuhaku.core.resolve.hardware import VRAM_HEADROOM, recommended_size_class  # noqa: E402
from kuhaku.tools.rag.resolve.store_policy import (  # noqa: E402
    CHUNK_UPGRADE_THRESHOLD,
    guard_single_writer,
    suggest_store_upgrade,
)

from tests.resolve.conftest import FakeAdapter, FakeUI, make_candidate, make_env


def test_vram_headroom_default_is_the_documented_estimate():
    assert VRAM_HEADROOM == 0.25


@pytest.mark.parametrize(
    ("vram_class", "expected"),
    [("none", "unknown"), ("unknown", "unknown"), ("small", "small"),
     ("medium", "medium"), ("large", "large")],
)
def test_recommended_size_class(vram_class, expected):
    assert recommended_size_class(vram_class) == expected


def test_recommended_size_class_reads_the_headroom_argument():
    # a punishing headroom shrinks the budget enough to drop a class
    assert recommended_size_class("medium", headroom=0.9) != recommended_size_class("medium", headroom=0.0)


def test_chunk_threshold_default_is_the_documented_estimate():
    assert CHUNK_UPGRADE_THRESHOLD == 50_000


def test_suggest_store_upgrade_reads_the_threshold_argument():
    reg = Registry()
    reg.register(FakeAdapter("store", [make_candidate("qdrant", "store")]))
    ui = FakeUI(interactive=True, ask_returns=None)
    # default threshold: silent
    suggest_store_upgrade(chunk_count=1_000, current_id="builtin", registry=reg,
                          env=make_env(), ui=ui)
    assert ui.ask_calls == []
    # lowered threshold: fires
    suggest_store_upgrade(chunk_count=1_000, current_id="builtin", registry=reg,
                          env=make_env(), ui=ui, threshold=500)
    assert len(ui.ask_calls) == 1


def test_suggest_store_upgrade_announces_no_reprocessing_on_an_explicit_pick():
    reg = Registry()
    reg.register(FakeAdapter("store", [make_candidate("qdrant", "store", label="Qdrant")]))
    ui = FakeUI(interactive=True, ask_returns="qdrant")
    choice = suggest_store_upgrade(chunk_count=99_999, current_id="builtin", registry=reg,
                                   env=make_env(), ui=ui)
    assert choice.id == "qdrant"
    assert any("reprocess" in m.lower() for m in ui.messages)


def test_guard_single_writer_second_holder_gets_store_conflict(tmp_path):
    ui = FakeUI(interactive=False)
    with guard_single_writer(tmp_path, ui=ui):
        with pytest.raises(StoreConflict):
            with guard_single_writer(tmp_path, ui=ui):
                pass
    # lock released on exit -> a later writer succeeds
    with guard_single_writer(tmp_path, ui=ui):
        pass
