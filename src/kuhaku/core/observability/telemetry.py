"""OpenTelemetry initialization: the single telemetry standard for kuhaku (D57).

``init_telemetry()`` installs the process-wide tracer/meter providers exactly once and is
called automatically at the bottom of this module -- importing
``kuhaku.core.observability`` (which every metrics/tracing call site already does) is
enough, no caller has to remember a manual setup step.

Zero-config defaults:

* Tracing: a real SDK ``TracerProvider`` is always installed (so spans are created
  in-process -- real trace/span IDs, real parent/child nesting, attributes work, which is
  what makes ``core.observability.tracing`` testable and lets
  ``core.llm.token_tracking.TokenTrackingLLM`` annotate "the current span"), but *no span
  processor is attached by default* -- nothing a span records is ever exported anywhere:
  no network calls, no background threads. Set ``OTEL_TRACES_EXPORTER=console`` or
  ``otlp`` (standard OTel env vars, not a new ``Settings`` field) to opt into export.
* Metrics: an SDK ``MeterProvider`` with a Prometheus exporter is always installed --
  collection is always on, exposition is gated at the embedding application's HTTP
  layer (e.g. an authenticated metrics route), not here, so this preserves
  that existing invariant. ``PrometheusMetricReader`` registers its own collector into
  ``prometheus_client``'s default ``REGISTRY`` (verified against the installed package),
  so ``generate_latest()`` at that route keeps working completely unmodified.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from prometheus_client import REGISTRY as _PROMETHEUS_REGISTRY

logger = logging.getLogger(__name__)

_SERVICE_NAME_DEFAULT = "kuhaku"

_AttributeValue = str | bool | int | float

# prometheus_client's own Histogram.DEFAULT_BUCKETS. OTel's SDK default bucket boundaries
# (0, 5, 10, 25 ... 10000) assume millisecond-scale measurements and would collapse every
# second-scale duration kuhaku records into a single bucket -- reproducing the
# previous defaults here is what "preserve existing metric values and meanings" requires
# for a duration histogram exported through the Prometheus bridge, not a new feature.
_DURATION_BUCKETS_SECONDS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
)

# Every duration Histogram kuhaku defines (core/observability/metrics.py) --
# kept as an explicit list (not a name pattern) so a future histogram must opt in
# deliberately rather than silently inheriting these bucket boundaries.
_DURATION_HISTOGRAM_NAMES = (
    "kuhaku_stage_duration_seconds",
    "api_request_duration_seconds",
    "kuhaku_evaluation_run_duration_seconds",
)

_lock = threading.Lock()
_initialized = False


def _build_duration_views() -> list[View]:
    return [
        View(
            instrument_name=name,
            aggregation=ExplicitBucketHistogramAggregation(boundaries=_DURATION_BUCKETS_SECONDS),
        )
        for name in _DURATION_HISTOGRAM_NAMES
    ]


def _configure_tracing(resource: Resource) -> None:
    provider = TracerProvider(resource=resource)
    exporter_name = os.environ.get("OTEL_TRACES_EXPORTER", "none").strip().lower()

    if exporter_name in ("", "none"):
        pass  # Zero-config default: spans are created in-process, never exported.
    elif exporter_name == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter_name == "otlp":
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    else:
        logger.warning(
            "Unknown OTEL_TRACES_EXPORTER=%r; tracing stays enabled but unexported "
            "(same as the default) -- expected 'none', 'console', or 'otlp'.",
            exporter_name,
        )

    otel_trace.set_tracer_provider(provider)


def _configure_metrics(resource: Resource) -> None:
    reader = PrometheusMetricReader()
    provider = MeterProvider(
        resource=resource, metric_readers=[reader], views=_build_duration_views()
    )
    otel_metrics.set_meter_provider(provider)


def init_telemetry() -> None:
    """Idempotently install the process-wide tracer/meter providers.

    Safe to call more than once (only the first call has any effect) -- guarded the same
    way ``metrics_summary.py``'s cache lock is, for the same reason (cheap, correct under
    concurrent first-import from multiple threads).
    """

    global _initialized
    with _lock:
        if _initialized:
            return
        resource = Resource.create(
            {SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", _SERVICE_NAME_DEFAULT)}
        )
        _configure_tracing(resource)
        _configure_metrics(resource)
        _initialized = True


def get_tracer() -> otel_trace.Tracer:
    """kuhaku's tracer.

    Bound lazily to whichever provider is currently installed (``opentelemetry.trace``'s
    ``ProxyTracer`` semantics) -- safe to call before or after ``init_telemetry()``, and
    safe for a test to swap the provider's span processors after the fact (see
    ``tests/observability/test_tracing.py``).
    """

    return otel_trace.get_tracer(_SERVICE_NAME_DEFAULT)


def get_meter() -> otel_metrics.Meter:
    """kuhaku's meter -- every instrument in ``metrics.py`` is created from this."""

    return otel_metrics.get_meter(_SERVICE_NAME_DEFAULT)


def collect_prometheus_families() -> list:
    """Read every registered Prometheus metric family from the default ``REGISTRY``.

    This module is kuhaku's one deliberate direct dependency on
    ``prometheus_client`` (see ``pyproject.toml``'s D57 note); everything else --
    ``metrics_summary.py``'s registry read included -- goes through this function rather
    than importing ``prometheus_client`` itself. Reading it here (not scraping the
    external ``/metrics`` HTTP port) works even when exposition is disabled, since
    ``_configure_metrics``'s ``PrometheusMetricReader`` registers its own collector into
    this exact ``REGISTRY`` regardless -- the OTel SDK has no synchronous "current value"
    read-back API of its own, so this is the one way to introspect current instrument
    values in-process.
    """

    return list(_PROMETHEUS_REGISTRY.collect())


class SettableGauge:
    """A synchronous-feeling ``.set(amount, attributes=None)`` gauge, backed by an OTel
    ``ObservableGauge`` rather than a plain synchronous ``Gauge``.

    The installed ``opentelemetry-exporter-prometheus``'s ``PrometheusMetricReader``
    hardcodes its ``preferred_temporality`` map (verified against the installed package's
    source: ``opentelemetry/exporter/prometheus/__init__.py``) to mark ``Counter``,
    ``UpDownCounter``, ``Histogram``, and ``ObservableGauge`` cumulative -- but *not* the
    newer synchronous ``Gauge`` API. A synchronous Gauge's value is recorded correctly but
    silently stops appearing in the Prometheus bridge after the collection cycle right
    after it was set, instead of persisting on every scrape the way a Prometheus Gauge
    always has (verified empirically: a second ``REGISTRY.collect()`` with no intervening
    ``.set()`` call returns no data point at all for a synchronous Gauge). An
    ``ObservableGauge`` callback sidesteps this: it reports whatever this class's
    internal state currently holds on every collection, indefinitely.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._values: dict[tuple[tuple[str, _AttributeValue], ...], float] = {}

    def set(
        self, amount: float, attributes: Mapping[str, _AttributeValue] | None = None
    ) -> None:
        key = tuple(sorted((attributes or {}).items()))
        with self._lock:
            self._values[key] = amount

    def _callback(self, options: CallbackOptions) -> list[Observation]:  # noqa: ARG002
        with self._lock:
            snapshot = list(self._values.items())
        return [Observation(value, dict(key)) for key, value in snapshot]


def create_settable_gauge(name: str, *, unit: str = "", description: str = "") -> SettableGauge:
    """Create a :class:`SettableGauge` -- the "current value" gauge every ``metrics.py``
    Gauge uses (see that class's docstring for why a plain ``meter.create_gauge()`` isn't
    used instead)."""

    gauge = SettableGauge(name)
    get_meter().create_observable_gauge(
        name, callbacks=[gauge._callback], unit=unit, description=description
    )
    return gauge


init_telemetry()
