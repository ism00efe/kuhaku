"""In-process Prometheus registry summarization for the admin metrics dashboard.

Reads the current Prometheus metric families via ``telemetry.collect_prometheus_families``
-- never scrapes the external ``/metrics`` HTTP port -- so this works even when
``Settings.metrics_enabled=False`` (the exposition server is optional; the underlying OTel
instruments always collect, per ``metrics.py``'s own module docstring). ``telemetry.py`` is
kuhaku's one deliberate direct dependency on ``prometheus_client``; this
module goes through its ``collect_prometheus_families`` wrapper rather than importing
``prometheus_client`` itself, so that dependency stays contained to one place. A separate
reader/aggregator module rather than growing ``metrics.py`` (pure metric *definitions*) --
the same split as ``metrics.py`` vs ``logging_context.py`` within this package.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from .telemetry import collect_prometheus_families

logger = logging.getLogger(__name__)

# Guards interleaved reads/writes from FastAPI's threadpool-dispatched sync handlers --
# same justification as security/audit.py's _write_lock. The rebuild itself runs OUTSIDE
# the lock (registry collection is read-only/thread-safe on its own); the lock only guards
# the cache dict's read/check/write, so a benign race on simultaneous expiry (two threads
# both rebuild) is possible and harmless -- both computations are idempotent and land
# within the same freshness window.
_cache_lock = threading.Lock()
_cache: dict[str, object] | None = None
_cache_at: float = 0.0
_CACHE_TTL_SECONDS = 5.0

_EMPTY_SUMMARY: dict[str, object] = {
    "request_count": 0,
    "error_count": 0,
    "error_rate": None,
    "cache_hits": 0,
    "cache_misses": 0,
    "cache_hit_ratio": None,
    "guard_reject_count": 0,
    "guard_total_count": 0,
    "guard_reject_rate": None,
    "avg_response_time_seconds": None,
}


def _family_by_name() -> dict[str, object]:
    return {family.name: family for family in collect_prometheus_families()}


def _counter_total(family: object, *, label_filter: Callable[[dict], bool] | None = None) -> float:
    """Sum a Counter family's value samples.

    Must filter to samples named exactly ``f"{family.name}_total"`` -- prometheus_client
    also emits a ``f"{family.name}_created"`` sample per label combination (a Unix-epoch
    creation timestamp, NOT a count). Naively summing every sample in the family would
    silently corrupt the total by adding in epoch-scale numbers. Verified directly against
    this repo's installed prometheus_client version rather than assumed.
    """

    if family is None:
        return 0.0
    total_name = f"{family.name}_total"  # type: ignore[attr-defined]
    return sum(
        s.value
        for s in family.samples  # type: ignore[attr-defined]
        if s.name == total_name and (label_filter is None or label_filter(s.labels))
    )


def _histogram_sum_count(family: object) -> tuple[float, float]:
    if family is None:
        return 0.0, 0.0
    sum_name = f"{family.name}_sum"  # type: ignore[attr-defined]
    count_name = f"{family.name}_count"  # type: ignore[attr-defined]
    total_sum = sum(s.value for s in family.samples if s.name == sum_name)  # type: ignore[attr-defined]
    total_count = sum(s.value for s in family.samples if s.name == count_name)  # type: ignore[attr-defined]
    return total_sum, total_count


def build_metrics_summary() -> dict[str, object]:
    """Compute a fresh summary. Never raises -- a missing/never-incremented family (an
    empty ``.samples`` list, or the family absent entirely) degrades every derived field
    to 0/``None`` rather than a ``KeyError``/``ZeroDivisionError``.
    """

    try:
        families = _family_by_name()

        api = families.get("api_requests")
        request_count = _counter_total(api)
        error_count = _counter_total(api, label_filter=lambda labels: labels.get("outcome") != "ok")
        error_rate = (error_count / request_count) if request_count else None

        hits = _counter_total(families.get("rag_cache_hits"))
        misses = _counter_total(families.get("rag_cache_misses"))
        cache_hit_ratio = (hits / (hits + misses)) if (hits + misses) else None

        zone = families.get("kuhaku_guard_zone")
        guard_reject = _counter_total(
            zone, label_filter=lambda labels: labels.get("zone") == "reject"
        )
        guard_total = _counter_total(zone)
        guard_reject_rate = (guard_reject / guard_total) if guard_total else None

        dur_sum, dur_count = _histogram_sum_count(families.get("api_request_duration_seconds"))
        avg_response_time = (dur_sum / dur_count) if dur_count else None

        return {
            "request_count": int(request_count),
            "error_count": int(error_count),
            "error_rate": error_rate,
            "cache_hits": int(hits),
            "cache_misses": int(misses),
            "cache_hit_ratio": cache_hit_ratio,
            "guard_reject_count": int(guard_reject),
            "guard_total_count": int(guard_total),
            "guard_reject_rate": guard_reject_rate,
            "avg_response_time_seconds": avg_response_time,
        }
    except Exception:
        logger.exception("failed to build metrics summary")
        return dict(_EMPTY_SUMMARY)


def get_cached_metrics_summary() -> dict[str, object]:
    """5-second cached summary (task requirement: cache for 5 seconds, simple in-memory
    dict with timestamp)."""

    global _cache, _cache_at
    now = time.monotonic()
    with _cache_lock:
        if _cache is not None and (now - _cache_at) < _CACHE_TTL_SECONDS:
            return _cache

    summary = build_metrics_summary()
    summary["generated_at"] = datetime.now(UTC).isoformat(timespec="milliseconds")

    with _cache_lock:
        _cache = summary
        _cache_at = now
    return summary
