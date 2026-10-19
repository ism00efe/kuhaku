"""Progress reporting for a batch run: the snapshot passed to the hook, and a default
plain-text reporter.

The hook is always called from the thread that called ``run()``, one item at a time, so
a reporter never needs its own locking even when items run concurrently.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from ..models import ExecutionResult


@dataclass(frozen=True)
class BatchProgress:
    """Where a run stands right after one item finished.

    ``failed`` is the number of items that ended unsuccessfully so far.
    """

    done: int
    total: int
    failed: int
    elapsed_seconds: float
    index: int
    item_id: str
    result: ExecutionResult


class TextProgressReporter:
    """Prints one line per finished item (or every ``every`` items, and always the last).

    ``[12/40] failed=1 elapsed=31.2s eta=72.8s``. Item ids are deliberately not printed:
    they are caller data and may carry content that should not end up in a CI log.
    """

    def __init__(self, stream: TextIO | None = None, *, every: int = 1) -> None:
        if every < 1:
            raise ValueError("every must be >= 1")
        self._stream = stream
        self._every = every

    def __call__(self, progress: BatchProgress) -> None:
        if progress.done % self._every and progress.done != progress.total:
            return
        eta = ""
        if progress.done and progress.done < progress.total:
            remaining = progress.elapsed_seconds / progress.done * (progress.total - progress.done)
            eta = f" eta={remaining:.1f}s"
        stream = self._stream if self._stream is not None else sys.stderr
        print(
            f"[{progress.done}/{progress.total}] failed={progress.failed} "
            f"elapsed={progress.elapsed_seconds:.1f}s{eta}",
            file=stream,
            flush=True,
        )
