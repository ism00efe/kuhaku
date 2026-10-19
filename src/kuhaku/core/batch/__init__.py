"""Batch execution: run one function over many items, concurrently.

Tool-agnostic -- the function is opaque to this package. A tool (or a script) supplies
``fn(item)``; ``BatchRunner`` supplies the thread pool, input-order results as
``ExecutionResult``, per-item trace ids and spans, retry, per-attempt timeouts, a
limiter seam and progress reporting.
"""

from .limiter import Limiter
from .progress import BatchProgress, TextProgressReporter
from .runner import BatchItemTimeout, BatchRunner, RetryPolicy, run_batch

__all__ = [
    "BatchItemTimeout",
    "BatchProgress",
    "BatchRunner",
    "Limiter",
    "RetryPolicy",
    "TextProgressReporter",
    "run_batch",
]
