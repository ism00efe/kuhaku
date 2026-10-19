"""``BatchRunner``: run one function over many items, concurrently, without losing any.

Tool-agnostic by construction -- it knows nothing about what ``fn`` does, only that it
takes one item and returns something or raises. Every item's outcome comes back as a
:class:`~kuhaku.core.models.ExecutionResult`, in input order, so one bad item never costs
the rest of the batch (unless ``fail_fast`` asks for exactly that).

Threads, not asyncio: every provider in ``core.llm`` is synchronous and I/O-bound, so a
thread pool gets the concurrency without a second, async provider stack.

Per-attempt stack, outermost first: retry (``core.retry.call_with_retry``) -> limiter ->
timeout -> ``fn(item)``. Retry is outermost so each retry re-acquires the limiter (a
rate limiter must see every provider call), and the limiter is acquired *outside* the
timeout so time spent waiting for a permit is never mistaken for a hung call.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar

from opentelemetry.trace import Status, StatusCode
from tenacity import retry_if_exception_type
from tenacity.wait import wait_base, wait_exponential

from ..exceptions import KuhakuError
from ..models import ExecutionResult
from ..observability import bind_trace_id, instrumented_step, new_trace_id, start_span
from ..retry import call_with_retry
from .limiter import check_limiter, hold
from .progress import BatchProgress

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

_CANCELLED_ERROR = "cancelled: an earlier item failed and fail_fast is set"


class BatchItemTimeout(KuhakuError, TimeoutError):
    """One attempt at an item ran past ``timeout_seconds``.

    The attempt is abandoned, not stopped -- Python cannot kill a thread, so the call
    keeps running in the background until it returns on its own, and its result is
    discarded. A provider's own HTTP timeout is what actually frees the connection.
    """


@dataclass(frozen=True)
class RetryPolicy:
    """Retry an item's call on exception, with backoff between attempts.

    ``max_attempts`` counts the first try. ``retry_on`` narrows which exceptions are
    worth retrying -- a validation error will fail identically every time.
    """

    max_attempts: int = 3
    wait: wait_base = field(default_factory=lambda: wait_exponential(multiplier=1, max=30))
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


class BatchRunner(Generic[T, R]):
    """Runs ``fn`` over a sequence of items; see the module docstring for the stack.

    Constructed once with the execution policy, then ``run()`` per batch -- the same
    runner can drive several datasets or functions.
    """

    def __init__(
        self,
        *,
        concurrency: int = 1,
        retry: RetryPolicy | None = None,
        timeout_seconds: float | None = None,
        limiter: object | None = None,
        on_progress: Callable[[BatchProgress], None] | None = None,
        fail_fast: bool = False,
        name: str = "batch",
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        check_limiter(limiter, cross_thread_release=timeout_seconds is not None)
        self._concurrency = concurrency
        self._retry = retry
        self._timeout = timeout_seconds
        self._limiter = limiter
        self._on_progress = on_progress
        self._fail_fast = fail_fast
        self._name = name

    def run(
        self,
        items: Iterable[T],
        fn: Callable[[T], R],
        *,
        key: Callable[[T], str] | None = None,
    ) -> list[ExecutionResult]:
        """Run ``fn`` over ``items``; one :class:`ExecutionResult` per item, input order.

        ``key`` gives each item a stable id (default: its position), used in progress
        reports and logs.
        """

        item_list: Sequence[T] = list(items)
        ids = [key(item) if key else str(i) for i, item in enumerate(item_list)]
        if key is not None and len(set(ids)) != len(ids):
            dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
            raise ValueError(f"key= produced duplicate ids: {dupes[:5]}")

        results: list[ExecutionResult | None] = [None] * len(item_list)
        todo = list(range(len(item_list)))
        state = _RunState(total=len(todo))

        def finish(index: int, result: ExecutionResult) -> None:
            # A failing progress callback is the caller's bookkeeping, not the item's
            # outcome: it must not abort the run (in pooled mode that abandons every
            # in-flight item unrecorded). It is reported on the item's result instead,
            # and logged by exception type only.
            state.done += 1
            if not result.success:
                state.failed += 1
            problems: dict[str, str] = {}
            results[index] = result
            if self._on_progress is not None:
                try:
                    self._on_progress(
                        BatchProgress(
                            done=state.done,
                            total=state.total,
                            failed=state.failed,
                            elapsed_seconds=time.perf_counter() - state.start,
                            index=index,
                            item_id=ids[index],
                            result=result,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- isolated and reported
                    problems["progress_error"] = type(exc).__name__
                    results[index] = replace(
                        result, metadata={**result.metadata, **problems}
                    )
            for kind, name in problems.items():
                logger.warning("batch item %d: %s (%s)", index, kind, name)

        with bind_trace_id(), instrumented_step(
            self._name,
            logger,
            **{
                "batch.items": len(item_list),
                "batch.concurrency": self._concurrency,
            },
        ) as step:
            if self._concurrency == 1 or len(todo) <= 1:
                self._run_sequential(todo, item_list, ids, fn, finish)
            else:
                self._run_pooled(todo, item_list, ids, fn, finish)
            step.set(
                result_count=state.done,
                **{"batch.failed": state.failed},
            )

        return [r if r is not None else _cancelled(i, ids[i]) for i, r in enumerate(results)]

    # --- scheduling ----------------------------------------------------------------

    def _run_sequential(
        self,
        todo: list[int],
        items: Sequence[T],
        ids: list[str],
        fn: Callable[[T], R],
        finish: Callable[[int, ExecutionResult], None],
    ) -> None:
        for index in todo:
            result = self._run_item(index, ids[index], items[index], fn)
            finish(index, result)
            if self._fail_fast and not result.success:
                return

    def _run_pooled(
        self,
        todo: list[int],
        items: Sequence[T],
        ids: list[str],
        fn: Callable[[T], R],
        finish: Callable[[int, ExecutionResult], None],
    ) -> None:
        pool = ThreadPoolExecutor(
            max_workers=min(self._concurrency, len(todo)), thread_name_prefix="kuhaku-batch"
        )
        try:
            # A fresh context copy per item: worker threads do not inherit contextvars,
            # and the open "batch" span and trace id live in them. One Context object
            # cannot be entered by two threads at once, hence one copy each.
            futures: dict[Future[ExecutionResult], int] = {
                pool.submit(
                    contextvars.copy_context().run,
                    self._run_item,
                    index,
                    ids[index],
                    items[index],
                    fn,
                ): index
                for index in todo
            }
            aborted = False
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                result = future.result()
                finish(futures[future], result)
                if self._fail_fast and not result.success and not aborted:
                    aborted = True
                    # Items already running finish and are recorded; only queued
                    # ones are dropped.
                    for pending in futures:
                        pending.cancel()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

    # --- one item ------------------------------------------------------------------

    def _run_item(self, index: int, item_id: str, item: T, fn: Callable[[T], R]) -> ExecutionResult:
        attempts = 0
        start = time.perf_counter()

        def attempt() -> R:
            nonlocal attempts
            attempts += 1
            return self._attempt(fn, item)

        with bind_trace_id(new_trace_id()) as trace_id, start_span(
            f"{self._name}.item", **{"batch.index": index}
        ) as span:
            metadata: dict[str, Any] = {"item_id": item_id, "index": index, "trace_id": trace_id}
            try:
                output = self._with_retry(attempt)
            except Exception as exc:  # noqa: BLE001 -- capturing every failure is the contract
                metadata.update(attempts=attempts, duration_seconds=_since(start))
                span.add_event("exception", {"exception.type": type(exc).__name__})
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("batch.attempts", attempts)
                # Exception type only: the message may quote item content.
                logger.warning(
                    "batch item %d failed after %d attempt(s): %s",
                    index,
                    attempts,
                    type(exc).__name__,
                    extra={"status": "error"},
                )
                return ExecutionResult(
                    output=None,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=metadata,
                )
            span.set_attribute("batch.attempts", attempts)
            metadata.update(attempts=attempts, duration_seconds=_since(start))
            return ExecutionResult(output=output, metadata=metadata)

    def _with_retry(self, attempt: Callable[[], R]) -> R:
        if self._retry is None:
            return attempt()
        return call_with_retry(
            attempt,
            service="batch",
            enabled=True,
            max_attempts=self._retry.max_attempts,
            wait=self._retry.wait,
            retry=retry_if_exception_type(self._retry.retry_on),
        )

    def _attempt(self, fn: Callable[[T], R], item: T) -> R:
        if self._timeout is None:
            with hold(self._limiter):
                return fn(item)

        # The permit is taken here, outside the timed window, but released by whichever
        # side finishes last -- an abandoned call still holds its slot until it returns,
        # because it is still using the provider.
        permit = ExitStack()
        permit.enter_context(hold(self._limiter))
        return _call_with_timeout(lambda: fn(item), self._timeout, on_exit=permit.close)


@dataclass
class _RunState:
    total: int
    done: int = 0
    failed: int = 0
    start: float = field(default_factory=time.perf_counter)


def _since(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def _cancelled(index: int, item_id: str) -> ExecutionResult:
    return ExecutionResult(
        output=None,
        success=False,
        error=_CANCELLED_ERROR,
        metadata={"item_id": item_id, "index": index, "cancelled": True},
    )


def _call_with_timeout(
    call: Callable[[], R], timeout: float, *, on_exit: Callable[[], object]
) -> R:
    box: dict[str, Any] = {}
    finished = threading.Event()
    context = contextvars.copy_context()

    def target() -> None:
        try:
            box["value"] = context.run(call)
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the waiting side
            box["error"] = exc
        finally:
            # finished.set() must run even when releasing the permit fails: otherwise a
            # call that succeeded is reported as a timeout. The call's own outcome
            # stands; the failed release is logged (the permit may have leaked).
            try:
                on_exit()
            except BaseException as exc:  # noqa: BLE001 -- must not mask the outcome
                logger.warning(
                    "batch limiter release failed (%s); the permit may have leaked",
                    type(exc).__name__,
                )
            finally:
                finished.set()

    threading.Thread(target=target, daemon=True, name="kuhaku-batch-timeout").start()
    if not finished.wait(timeout):
        raise BatchItemTimeout(f"attempt exceeded {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]  # type: ignore[no-any-return]


def run_batch(
    items: Iterable[T],
    fn: Callable[[T], R],
    *,
    concurrency: int = 1,
    key: Callable[[T], str] | None = None,
    retry: RetryPolicy | None = None,
    timeout_seconds: float | None = None,
    limiter: object | None = None,
    on_progress: Callable[[BatchProgress], None] | None = None,
    fail_fast: bool = False,
) -> list[ExecutionResult]:
    """One-shot form of ``BatchRunner(...).run(items, fn, ...)``."""

    runner: BatchRunner[T, R] = BatchRunner(
        concurrency=concurrency,
        retry=retry,
        timeout_seconds=timeout_seconds,
        limiter=limiter,
        on_progress=on_progress,
        fail_fast=fail_fast,
    )
    return runner.run(items, fn, key=key)
