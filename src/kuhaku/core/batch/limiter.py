"""The limiter seam: the minimal protocol a rate limiter or concurrency gate plugs into.

``BatchRunner`` does not ship a limiter -- it only defines where one goes. The limiter is
held once per *attempt*, so a retried item pays for every provider call it actually
makes. Two shapes are accepted because the two common limiter designs need different
things:

- a *token bucket* only needs to block until a permit is available; nothing is handed
  back afterwards, so its ``acquire()`` blocks and returns ``None`` (or any value that
  is not a context manager -- it is ignored);
- a *concurrency gate* (an in-flight budget) must release when the call ends, so its
  ``acquire()`` returns a context manager held for exactly the call.

An object that is itself a context manager is always used with ``with`` and its
``acquire()`` (if any) is not called directly. That rule is what makes a bare
``threading.Semaphore`` work: its ``acquire()`` returns ``True`` and needs a matching
``release()``, which only its ``__exit__`` provides.

With a per-attempt timeout the permit is released on the worker thread that ran the
call, not the one that acquired it, so a limiter must be releasable from any thread.
``threading.Semaphore``/``BoundedSemaphore`` and ``Lock`` are; an ``RLock`` is not (only
its owner may release it) and is refused when a timeout is set.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Limiter(Protocol):
    """Blocks in ``acquire()`` until the caller may proceed.

    Return ``None`` when nothing needs releasing, or a context manager that is entered
    immediately and exited when the guarded call finishes (success or failure).
    """

    def acquire(self) -> AbstractContextManager[Any] | None: ...


def _is_context_manager(obj: object) -> bool:
    return hasattr(obj, "__enter__") and hasattr(obj, "__exit__")


@contextmanager
def hold(limiter: object | None) -> Iterator[None]:
    """Hold ``limiter`` (``None`` = no limit) around one guarded call."""

    if limiter is None:
        yield
    elif _is_context_manager(limiter):
        with limiter:  # type: ignore[attr-defined]
            yield
    elif callable(getattr(limiter, "acquire", None)):
        permit = limiter.acquire()  # type: ignore[attr-defined]
        if _is_context_manager(permit):
            with permit:
                yield
        else:
            yield
    else:
        raise TypeError(
            "limiter must have an acquire() method or be a context manager, got "
            f"{type(limiter).__name__}"
        )


_RLOCK_TYPES = tuple({type(threading.RLock()), threading._RLock})  # type: ignore[attr-defined]


def check_limiter(limiter: object | None, *, cross_thread_release: bool = False) -> None:
    """Fail at construction, not on the first item, for an object that cannot be held.

    ``cross_thread_release`` (a timeout is set): refuse an ``RLock``, which raises when
    released by a thread other than its owner -- the permit would leak.
    """

    if cross_thread_release and isinstance(limiter, _RLOCK_TYPES):
        raise TypeError(
            "an RLock cannot be a batch limiter with timeout_seconds: the permit is "
            "released on a different thread than the one that acquired it. Use a "
            "threading.Semaphore or BoundedSemaphore."
        )
    if limiter is None or _is_context_manager(limiter):
        return
    if not callable(getattr(limiter, "acquire", None)):
        raise TypeError(
            "limiter must have an acquire() method or be a context manager, got "
            f"{type(limiter).__name__}"
        )
