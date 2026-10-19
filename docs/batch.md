# Batch runs

`kuhaku.core.batch` runs one function over many items: concurrently, with every item's
outcome kept. It knows nothing about what the
function does — an extraction call, a RAG question, an evaluation row — so any tool or
script can use it.

```python
from kuhaku.core.batch import run_batch


def classify(text: str) -> str:
    if not text:
        raise ValueError("empty input")
    return "long" if len(text) > 10 else "short"


results = run_batch(["hello", "", "a much longer text"], classify, concurrency=4)

for result in results:
    print(result.success, result.output, result.error)
# True short None
# False None ValueError: empty input
# True long None
```

Each item comes back as an `ExecutionResult` (`kuhaku.core.models`), **in input order**,
regardless of which finished first. An exception is captured into that item's result
(`success=False`, `error="ExceptionType: message"`) and the rest of the batch carries on.
`metadata` holds `item_id`, `index`, `trace_id`, `attempts` and `duration_seconds`.

Concurrency is a thread pool. kuhaku's LLM providers are synchronous and spend their time
waiting on the network, so threads are what gets the speed-up; your function must be safe
to call from several threads at once. `concurrency=1` (the default) runs the items one
after another on the calling thread.

## The reusable form

`run_batch(...)` is shorthand for building a `BatchRunner` and calling `run()` once. A
runner holds the execution policy and can be reused for several batches:

```python
from tenacity import wait_none

from kuhaku.core.batch import BatchRunner, RetryPolicy

runner = BatchRunner(
    concurrency=8,
    retry=RetryPolicy(max_attempts=3, wait=wait_none(), retry_on=(ConnectionError,)),
    timeout_seconds=30,
)

print([r.output for r in runner.run([1, 2, 3], lambda n: n * n)])   # [1, 4, 9]
print([r.output for r in runner.run([4], lambda n: n * n)])         # [16]
```

| Option | Default | Effect |
|---|---|---|
| `concurrency` | `1` | worker threads |
| `retry` | `None` | a `RetryPolicy`: `max_attempts` (counting the first try), `wait` (any tenacity wait, default exponential up to 30 s), `retry_on` (exception types worth retrying) |
| `timeout_seconds` | `None` | per-attempt limit; an attempt over it fails with `BatchItemTimeout` |
| `limiter` | `None` | a rate limiter or concurrency gate, see below |
| `on_progress` | `None` | called after every finished item, see below |
| `fail_fast` | `False` | stop at the first failure |
| `name` | `"batch"` | span name for the run (`<name>.item` for each item) |

Retry goes through `kuhaku.core.retry.call_with_retry`, so retries show up in the
`kuhaku_retry_*` metrics under `service="batch"`.

**A timeout abandons the attempt, it does not stop it.** Python cannot kill a thread: the
timed-out call keeps running in the background until it returns by itself, and its result
is thrown away. `BatchItemTimeout` is a `TimeoutError`, so it is retried when `retry_on`
includes it (the default `(Exception,)` does). Keep the provider's own HTTP timeout set —
that is what actually frees the connection.

**`fail_fast=True`** stops at the first failed item. Items still queued are not run and
come back with `success=False`, `metadata["cancelled"] = True`; items already running when
the failure arrived finish and keep their real results. Nothing is raised.

## Progress

`on_progress` receives a `BatchProgress` after each finished item: `done`, `total`,
`failed`, `elapsed_seconds`, and the
item's `index`, `item_id` and `result`. It is always called on the thread that called
`run()`, one item at a time, so it needs no locking. `TextProgressReporter` is a ready-made
one that prints a line to stderr (or a stream you pass):

```python
import sys

from kuhaku.core.batch import TextProgressReporter, run_batch

run_batch(range(6), lambda n: n, on_progress=TextProgressReporter(sys.stdout, every=3))
# [3/6] failed=0 elapsed=0.0s eta=0.0s
# [6/6] failed=0 elapsed=0.0s
```

It prints counts only — never item ids, which are your data.

## Rate limiting

`limiter=` takes a rate limiter or a concurrency gate. The protocol is deliberately small
(`kuhaku.core.batch.Limiter`) so one can be plugged in:

- an object with `acquire()` that blocks until the call may proceed. If `acquire()`
  returns a context manager, it is held for the duration of the call (a concurrency
  gate); otherwise nothing is released afterwards (a token bucket);
- or any context manager, used with `with` — a plain `threading.Semaphore` works.

```python
import threading

from kuhaku.core.batch import run_batch

at_most_two_at_once = threading.Semaphore(2)
results = run_batch(range(10), lambda n: n + 1, concurrency=8, limiter=at_most_two_at_once)
print(sum(r.output for r in results))   # 55
```

kuhaku's own `RateLimiter` (see [Rate limits](limits.md)) fits the first
shape: passed as `limiter=`, it admits one request per item attempt and charges no
tokens, so it paces a run to `requests_per_minute`. To pace by tokens instead, wrap the
provider the items call in `RateLimitedLLM` with that limiter — not both, or each call is
counted as two requests.

```python
from kuhaku.core.batch import run_batch
from kuhaku.core.llm.limits import RateLimiter, RateLimits

limiter = RateLimiter(RateLimits(requests_per_minute=600))
results = run_batch(range(5), lambda n: n * 2, limiter=limiter)
print([r.output for r in results])   # [0, 2, 4, 6, 8]
```

The limiter is taken once per attempt, so each retry pays for its own provider call. It is
acquired before the timeout clock starts, so waiting for a permit never counts as a slow
call, and a timed-out call keeps its slot until it actually returns.

## Tracing

The run opens one span (`batch`, or the runner's `name`) and binds a trace id, inheriting
one already bound by the caller. Each item runs under its own span (`batch.item`, a child
of the run span even though it executes on a worker thread) and its own fresh trace id, so
the log lines of one item can be pulled out of an interleaved concurrent log. That id is
in the item's `metadata["trace_id"]`. A failed item's span is marked as an error. See
[Observability](observability.md).

## Concurrent evaluation

`EvaluationRunner.run()` accepts `concurrency=` and `on_progress=` and uses this runner
for them — see [Evaluation](evaluation.md#running-it).
