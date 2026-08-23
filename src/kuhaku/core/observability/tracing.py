"""Span helper + GenAI semantic-convention attribute keys.

``start_span`` is the one seam ``instrumented_step`` (``core/observability/__init__.py``)
uses to open an OTel span for every existing pipeline stage. It relies on the SDK's own
``start_as_current_span`` exception handling (records the exception, sets the span status
to ERROR, re-raises) rather than reimplementing it -- and never lets a bad attribute value
turn a tracing concern into a request failure: attribute-setting is wrapped in its own
``try/except`` (Feature 3's edge case).

GenAI attribute keys are still under ``opentelemetry.semconv._incubating`` in the installed
``opentelemetry-semantic-conventions`` package -- an explicitly private, no-compatibility-
guarantee import path. Defined here as plain string literals instead, matching the same
spec values, so this module (and anything importing these constants) isn't hostage to that
package's internal layout changing across minor versions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Span

from .telemetry import get_tracer

logger = logging.getLogger(__name__)

# --- GenAI semantic conventions (incubating upstream) -------------------------------
# Used by the "generate" pipeline stage span -- see core/llm/token_tracking.py, which
# annotates whatever span is ambient with these once a call completes.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

_AttributeValue = str | bool | int | float


def _coerce_attributes(attributes: Mapping[str, object]) -> dict[str, _AttributeValue]:
    """Drop ``None`` values; coerce anything not already an OTel-safe primitive.

    Every value kuhaku actually passes (stage names, counts, booleans, floats,
    token counts) is already a valid primitive -- this is a defensive floor, not the
    expected path.
    """

    coerced: dict[str, _AttributeValue] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        coerced[key] = value if isinstance(value, (str, bool, int, float)) else str(value)
    return coerced


def set_span_attributes(span: Span, attributes: Mapping[str, object]) -> None:
    """Best-effort: coerce and set ``attributes`` on ``span``.

    Shared by :func:`start_span` and :func:`set_current_span_attributes` (and by
    ``instrumented_step``'s combined log+span recorder in ``__init__.py``) so there is one
    place that decides "a bad attribute value must never break the caller" (Feature 3's
    edge case) rather than three copies of the same ``try/except``.
    """

    if not attributes:
        return
    try:
        span.set_attributes(_coerce_attributes(attributes))
    except Exception:
        logger.debug("failed to set span attributes", exc_info=True)


@contextmanager
def start_span(name: str, **attributes: object) -> Iterator[Span]:
    """Open an OTel span named ``name``, as a child of whatever span is current.

    On an exception raised inside the block, the SDK's own ``start_as_current_span``
    records it and marks the span ERROR before re-raising -- nothing extra needed here.
    """

    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        set_span_attributes(span, attributes)
        yield span


def set_current_span_attributes(**attributes: object) -> None:
    """Best-effort: set attributes on whatever span is currently open.

    Used by ``TokenTrackingLLM`` to attach GenAI token-usage attributes onto the ambient
    "generate" span ``RAGEngine`` already has open, without
    ``TokenTrackingLLM`` needing to know anything about span management itself -- same
    loosely-coupled "observer, not a protocol change" stance that class's module docstring
    already documents for metrics. A no-op when no span is current: this must never turn a
    successful LLM call into a failed request.
    """

    span = trace.get_current_span()
    if span.is_recording():
        set_span_attributes(span, attributes)
