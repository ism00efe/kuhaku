"""BatchRunner: a failing permit release, checkpoint write or progress callback is
isolated; checkpoints carry a fingerprint; RLock limiters are refused with a timeout."""

from __future__ import annotations

import logging
import threading

import pytest

from kuhaku.core.batch import (
    BatchRunner,
    run_batch,
)
from kuhaku.core.batch.runner import BatchItemTimeout
from kuhaku.core.exceptions import KuhakuError


class _ExplodingRelease:
    """A context-manager limiter whose release always raises."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        raise RuntimeError("release failed")


def test_release_failure_does_not_turn_success_into_timeout(caplog):
    with caplog.at_level(logging.WARNING, logger="kuhaku.core.batch.runner"):
        results = run_batch([1, 2], lambda n: n * 10, timeout_seconds=5, limiter=_ExplodingRelease())
    assert [r.success for r in results] == [True, True]
    assert [r.output for r in results] == [10, 20]
    assert any("limiter release failed" in r.getMessage() for r in caplog.records)


def test_rlock_limiter_refused_with_timeout():
    with pytest.raises(TypeError, match="RLock"):
        BatchRunner(timeout_seconds=1, limiter=threading.RLock())
    BatchRunner(limiter=threading.RLock())  # no timeout: same thread releases


def test_batch_item_timeout_is_a_kuhaku_error():
    assert issubclass(BatchItemTimeout, KuhakuError)
    assert issubclass(BatchItemTimeout, TimeoutError)


@pytest.mark.parametrize("concurrency", [1, 4])
def test_progress_callback_failure_is_isolated(concurrency):
    def on_progress(progress):
        if progress.item_id == "2":
            raise ValueError("ui broke")

    results = run_batch(range(5), lambda n: n, concurrency=concurrency, on_progress=on_progress)
    assert [r.output for r in results] == [0, 1, 2, 3, 4]
    assert results[2].metadata["progress_error"] == "ValueError"
    assert "progress_error" not in results[1].metadata


def _lower(text: str) -> str:
    return text.lower()


