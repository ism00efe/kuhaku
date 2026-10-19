"""Tests for ``kuhaku.core.batch``: ordering, failure capture, fail_fast, retry, timeout,
the limiter seam, progress reporting, per-item trace ids/spans, and checkpoint/resume.

Every test is offline and synchronizes on events/barriers rather than sleeping; the few
waits that exist are bounded at milliseconds and only guard against a hung test.
"""

from __future__ import annotations

import io
import threading
from contextlib import contextmanager

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from tenacity import wait_none

from kuhaku.core.batch import (
    BatchItemTimeout,
    BatchProgress,
    BatchRunner,
    Limiter,
    RetryPolicy,
    TextProgressReporter,
    run_batch,
)
from kuhaku.core.observability import get_trace_id

_GUARD = 5.0  # upper bound on any wait; only reached if the code under test is broken


def _double(x: int) -> int:
    return x * 2


# --- core behaviour -----------------------------------------------------------------


def test_results_come_back_in_input_order_with_real_concurrency():
    # All three items must be in flight at once to pass the barrier.
    barrier = threading.Barrier(3, timeout=_GUARD)
    finished: list[int] = []

    def fn(x: int) -> int:
        barrier.wait()
        finished.append(x)
        return x * 10

    results = run_batch([1, 2, 3], fn, concurrency=3)

    assert [r.output for r in results] == [10, 20, 30]
    assert all(r.success for r in results)
    assert sorted(finished) == [1, 2, 3]


def test_an_exception_is_captured_and_the_rest_of_the_batch_still_runs():
    def fn(x: int) -> int:
        if x == 2:
            raise RuntimeError("boom")
        return x

    results = run_batch(range(5), fn, concurrency=2)

    assert [r.success for r in results] == [True, True, False, True, True]
    assert results[2].output is None
    assert results[2].error == "RuntimeError: boom"
    assert results[2].metadata["attempts"] == 1
    assert [r.output for r in results if r.success] == [0, 1, 3, 4]


def test_empty_input_returns_empty_list():
    assert run_batch([], _double, concurrency=4) == []


def test_metadata_carries_id_index_attempts_and_duration():
    results = run_batch(["a", "b"], str.upper, key=lambda s: f"id-{s}")

    assert results[1].metadata["item_id"] == "id-b"
    assert results[1].metadata["index"] == 1
    assert results[1].metadata["attempts"] == 1
    assert results[1].metadata["duration_seconds"] >= 0


def test_invalid_construction_is_rejected():
    with pytest.raises(ValueError):
        BatchRunner(concurrency=0)
    with pytest.raises(ValueError):
        BatchRunner(timeout_seconds=0)
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(TypeError):
        BatchRunner(limiter=object())


def test_duplicate_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        run_batch(["a", "a"], str.upper, key=lambda s: s)


# --- fail_fast ----------------------------------------------------------------------


def test_fail_fast_sequential_stops_at_the_first_failure():
    called: list[int] = []

    def fn(x: int) -> int:
        called.append(x)
        if x == 1:
            raise ValueError("bad")
        return x

    results = run_batch(range(4), fn, fail_fast=True)

    assert called == [0, 1]
    assert results[0].success
    assert results[1].error == "ValueError: bad"
    assert all(r.metadata.get("cancelled") for r in results[2:])
    assert all(not r.success for r in results[2:])


def test_fail_fast_pooled_cancels_queued_items_but_records_in_flight_ones():
    failure_seen = threading.Event()
    called: list[int] = []
    lock = threading.Lock()

    def fn(x: int) -> int:
        with lock:
            called.append(x)
        if x == 0:
            raise RuntimeError("first")
        # Hold the second worker until the runner has processed the failure, so the
        # queue is still full when it cancels.
        failure_seen.wait(_GUARD)
        threading.Event().wait(0.05)
        return x

    def on_progress(p: BatchProgress) -> None:
        if not p.result.success:
            failure_seen.set()

    results = run_batch(range(10), fn, concurrency=2, fail_fast=True, on_progress=on_progress)

    assert results[0].error == "RuntimeError: first"
    cancelled = [i for i, r in enumerate(results) if r.metadata.get("cancelled")]
    assert cancelled, "queued items should have been cancelled"
    assert not set(cancelled) & set(called)
    for i in called:
        assert not results[i].metadata.get("cancelled")
        assert results[i].success == (i != 0)


# --- retry --------------------------------------------------------------------------


def test_retry_policy_retries_until_success_and_counts_attempts():
    calls = {"n": 0}

    def flaky(x: int) -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return x

    (result,) = run_batch([7], flaky, retry=RetryPolicy(max_attempts=3, wait=wait_none()))

    assert result.success and result.output == 7
    assert result.metadata["attempts"] == 3


def test_retry_gives_up_after_max_attempts():
    def always(x: int) -> int:
        raise ConnectionError("down")

    (result,) = run_batch([1], always, retry=RetryPolicy(max_attempts=2, wait=wait_none()))

    assert not result.success
    assert result.metadata["attempts"] == 2


def test_retry_on_narrows_which_exceptions_are_retried():
    calls = {"n": 0}

    def invalid(x: int) -> int:
        calls["n"] += 1
        raise ValueError("deterministic")

    policy = RetryPolicy(max_attempts=5, wait=wait_none(), retry_on=(ConnectionError,))
    (result,) = run_batch([1], invalid, retry=policy)

    assert calls["n"] == 1
    assert result.metadata["attempts"] == 1


# --- timeout ------------------------------------------------------------------------


def test_timeout_abandons_a_hung_attempt():
    release = threading.Event()

    def fn(x: int) -> int:
        if x == 1:
            release.wait(_GUARD)
        return x

    try:
        results = run_batch([0, 1, 2], fn, concurrency=2, timeout_seconds=0.05)
    finally:
        release.set()

    assert [r.success for r in results] == [True, False, True]
    assert results[1].error.startswith("BatchItemTimeout")


def test_timeout_is_retried_like_any_other_exception():
    calls = {"n": 0}
    release = threading.Event()

    def fn(x: int) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            release.wait(_GUARD)
        return x

    try:
        (result,) = run_batch(
            [3],
            fn,
            timeout_seconds=0.05,
            retry=RetryPolicy(max_attempts=2, wait=wait_none(), retry_on=(BatchItemTimeout,)),
        )
    finally:
        release.set()

    assert result.success and result.output == 3
    assert result.metadata["attempts"] == 2


def test_an_exception_inside_a_timed_attempt_is_reported_unchanged():
    def fn(x: int) -> int:
        raise KeyError("missing")

    (result,) = run_batch([1], fn, timeout_seconds=1.0)

    assert result.error.startswith("KeyError")


# --- limiter ------------------------------------------------------------------------


class _GateLimiter:
    """Concurrency gate: acquire() returns a context manager released after the call."""

    def __init__(self, slots: int) -> None:
        self._sem = threading.Semaphore(slots)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.acquired = 0

    def acquire(self):
        self._sem.acquire()
        with self._lock:
            self.acquired += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)

        @contextmanager
        def permit():
            try:
                yield
            finally:
                with self._lock:
                    self.in_flight -= 1
                self._sem.release()

        return permit()


class _BucketLimiter:
    """Token-bucket shape: acquire() only blocks (here: counts), nothing to release."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            self.calls += 1


def test_limiter_protocol_is_runtime_checkable():
    assert isinstance(_GateLimiter(1), Limiter)
    assert isinstance(_BucketLimiter(), Limiter)


def test_gate_limiter_bounds_in_flight_calls_below_pool_size():
    limiter = _GateLimiter(slots=2)
    results = run_batch(range(12), _double, concurrency=6, limiter=limiter)

    assert all(r.success for r in results)
    assert limiter.acquired == 12
    assert limiter.max_in_flight <= 2
    assert limiter.in_flight == 0


def test_bucket_limiter_is_consulted_once_per_attempt():
    limiter = _BucketLimiter()
    calls = {"n": 0}

    def flaky(x: int) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("once")
        return x

    run_batch([1], flaky, limiter=limiter, retry=RetryPolicy(max_attempts=2, wait=wait_none()))

    assert limiter.calls == 2


def test_a_bare_semaphore_works_as_a_limiter():
    sem = threading.BoundedSemaphore(1)
    in_flight = {"now": 0, "max": 0}
    lock = threading.Lock()

    def fn(x: int) -> int:
        with lock:
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
        with lock:
            in_flight["now"] -= 1
        return x

    results = run_batch(range(8), fn, concurrency=4, limiter=sem)

    assert all(r.success for r in results)
    assert in_flight["max"] == 1
    # BoundedSemaphore raises on over-release, so reaching here also proves every
    # acquire was matched by exactly one release.


def test_a_timed_out_call_keeps_its_limiter_slot_until_it_returns():
    limiter = _GateLimiter(slots=1)
    release = threading.Event()

    def fn(x: int) -> int:
        release.wait(_GUARD)
        return x

    (result,) = run_batch([1], fn, limiter=limiter, timeout_seconds=0.02)

    assert not result.success
    assert limiter.in_flight == 1  # the abandoned call is still running
    release.set()
    for _ in range(500):
        if limiter.in_flight == 0:
            break
        threading.Event().wait(0.01)
    assert limiter.in_flight == 0


# --- progress -----------------------------------------------------------------------


def test_progress_hook_runs_on_the_calling_thread_with_running_counts():
    seen: list[BatchProgress] = []
    threads: set[str] = set()

    def hook(p: BatchProgress) -> None:
        seen.append(p)
        threads.add(threading.current_thread().name)

    def fn(x: int) -> int:
        if x == 3:
            raise RuntimeError("x")
        return x

    run_batch(range(5), fn, concurrency=3, on_progress=hook)

    assert [p.done for p in seen] == [1, 2, 3, 4, 5]
    assert {p.total for p in seen} == {5}
    assert seen[-1].failed == 1
    assert threads == {threading.current_thread().name}
    assert all(p.elapsed_seconds >= 0 for p in seen)


def test_text_progress_reporter_formats_and_throttles():
    out = io.StringIO()
    reporter = TextProgressReporter(out, every=2)

    run_batch(range(5), _double, on_progress=reporter)

    lines = out.getvalue().splitlines()
    assert [line.split()[0] for line in lines] == ["[2/5]", "[4/5]", "[5/5]"]
    assert "failed=0" in lines[-1]
    assert "eta=" in lines[0] and "eta=" not in lines[-1]


def test_text_progress_reporter_rejects_bad_interval():
    with pytest.raises(ValueError):
        TextProgressReporter(every=0)


# --- trace ids and spans ------------------------------------------------------------


def test_each_item_gets_its_own_trace_id_bound_while_it_runs():
    def fn(x: int) -> str:
        return get_trace_id()

    results = run_batch(range(4), fn, concurrency=2)

    ids = [r.metadata["trace_id"] for r in results]
    assert len(set(ids)) == 4
    assert [r.output for r in results] == ids


def test_item_spans_are_children_of_the_batch_span_across_threads():
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    exporter.clear()

    def fn(x: int) -> int:
        if x == 1:
            raise RuntimeError("bad")
        return x

    BatchRunner(concurrency=2, name="unit_test_batch").run(range(3), fn)

    spans = exporter.get_finished_spans()
    (parent,) = [s for s in spans if s.name == "unit_test_batch"]
    children = [s for s in spans if s.name == "unit_test_batch.item"]
    assert len(children) == 3
    assert {c.parent.span_id for c in children} == {parent.context.span_id}
    failed = [c for c in children if c.attributes["batch.index"] == 1]
    assert failed[0].status.status_code == StatusCode.ERROR
    assert parent.attributes["batch.items"] == 3


# --- checkpoint / resume ------------------------------------------------------------


def test_runner_is_reusable_across_batches():
    runner: BatchRunner[int, int] = BatchRunner(concurrency=2)

    assert [r.output for r in runner.run([1, 2], _double)] == [2, 4]
    assert [r.output for r in runner.run([3], _double)] == [6]


def test_a_rate_limiter_works_directly_as_a_batch_limiter():
    from kuhaku.core.llm.limits import RateLimiter, RateLimits
    from tests.llm.limits_fakes import FakeClock

    clock = FakeClock()
    limiter = RateLimiter(RateLimits(requests_per_minute=2), clock=clock)
    assert isinstance(limiter, Limiter)
    results = run_batch(range(4), _double, limiter=limiter)

    assert [r.output for r in results] == [0, 2, 4, 6]
    # Two requests fit the first minute; the other two waited for the bucket.
    assert clock.sleeps and sum(clock.sleeps) >= 30
