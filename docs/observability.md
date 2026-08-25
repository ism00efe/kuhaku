# Observability

kuhaku emits structured logs, OpenTelemetry spans and Prometheus-compatible metrics. It
does **not** configure logging, start an HTTP server, or choose an exporter — those
belong to your application. This page shows what kuhaku produces and how to collect it.

## Trace ids

Every request gets a short trace id that ties its log lines, spans and audit record
together. It is on the answer:

```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data")
answer = rag.ask("How do I reset my password?")
print(answer.trace_id)      # e.g. "8f21a0c4d3b7"
```

The id is a 12-character hex value, generated per request and carried through a
`ContextVar` rather than passed as an argument.

To reuse your own request id instead — so kuhaku's records line up with your web
framework's — bind it around the call:

```python
from kuhaku import RAG
from kuhaku.core.observability.logging_context import bind_trace_id

rag = RAG(vector_store="./kuhaku-data")

with bind_trace_id("req-1234"):
    answer = rag.ask("How do I reset my password?")

print(answer.trace_id)      # "req-1234"
```

An id already bound by an outer scope is inherited; only a genuinely unbound context
mints a new one.

Note that this trace id is kuhaku's own and is independent of the OpenTelemetry trace id.
Nothing correlates the two automatically.

## Logging

kuhaku calls `logging.getLogger(__name__)` in each module and nothing else. No handler,
no level, no format — so out of the box its log lines follow whatever your application
configured, and **the trace id will not appear** until you attach the filter that injects
it.

The pieces are supplied; wiring them is one function:

<!-- no-exec -->
```python
import logging
from kuhaku.core.observability.logging_context import JsonFormatter, TraceIdFilter

def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(TraceIdFilter())      # must be on the handler, not the logger
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

configure_logging()
```

Each line then comes out as one JSON object with `timestamp`, `level`, `logger`,
`message` and `trace_id`, plus whatever structured fields the call site attached.

If you prefer your own format, keep `TraceIdFilter` and drop `JsonFormatter` — the filter
is what puts `record.trace_id` in place for a `%(trace_id)s` in your own format string.

## Spans

Each stage of a request opens a span. The names, in the order they can occur:

`answer` · `sanitize` · `input_guard` · `retrieve` · `embed` · `rerank` · `cache_lookup` ·
`generate` · `cite`

`answer` wraps the whole request and carries the `trace_id` as an attribute. `embed` and
`rerank` are opened inside retrieval. Token counts land on the `generate` span as
`gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens`, alongside `gen_ai.system`
and `gen_ai.request.model`.

**No span exporter is attached by default.** Telemetry initialises itself on import with
a real tracer provider but no processor, so spans are created and discarded. Turn on an
exporter with the standard OpenTelemetry environment variables — these are not kuhaku
settings:

```bash
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=my-service          # default: kuhaku
```

`OTEL_TRACES_EXPORTER=console` prints spans to stdout, which is the quickest way to see
whether instrumentation is reaching you at all.

## Metrics

Metrics are always collected — there is no switch — into the default `prometheus_client`
registry. The instruments a library user will actually see values on:

| Metric | Type | Labels |
|---|---|---|
| `kuhaku_stage_duration_seconds` | histogram | `stage` |
| `kuhaku_sanitization_redactions_total` | counter | `category` |
| `kuhaku_llm_tokens_total` | counter | `gen_ai.system`, `gen_ai.request.model`, `gen_ai.token.type` |
| `kuhaku_retry_attempts_total` | counter | `service` |
| `kuhaku_retry_successes_total` / `_failures_total` | counter | `service` |
| `kuhaku_audit_records_total` | counter | — |
| `rag_requests_total` | counter | `status` |
| `rag_retriever_strategy_total` | counter | `strategy` |
| `rag_abstention_total` | counter | `reason` |
| `rag_cache_hits_total` / `rag_cache_misses_total` | counter | — |
| `rag_active_documents_count` | gauge | — |
| `rag_unverified_citations_total` | counter | `citation` |

`rag_requests_total{status}` is the one to watch: `ok`, `empty`, `blocked`, `empty_kb`,
`no_chunks`. `rag_abstention_total{reason}` splits abstentions into `zero_chunks` and
`low_confidence`.

`service` on the retry counters is one of `llm`, `embedding`, `vectorstore`, `reranker`.

### Exposing them

kuhaku ships no HTTP endpoint. Serve the registry yourself:

<!-- no-exec -->
```python
from prometheus_client import start_http_server

start_http_server(9090)      # GET http://localhost:9090/metrics
```

Or inside an existing web app, expose `prometheus_client.generate_latest()` on a route.

`prometheus-client` arrives as a transitive dependency of the OpenTelemetry Prometheus
exporter, so it is already installed — but if you import it directly, declare it in your
own dependencies rather than relying on that.

### Reading a summary in-process

```python
from kuhaku.core.observability.metrics_summary import get_cached_metrics_summary

print(get_cached_metrics_summary())
```

Returns cache hit ratio, guard reject rate and error rate as plain numbers, cached for
five seconds. Fields derived from the HTTP request counters read zero for a library user,
because kuhaku has no HTTP layer to populate them.

## Token accounting

On by default. Each LLM call records token counts to `kuhaku_llm_tokens_total`, sets them
as span attributes, and emits an `llm usage` log line.

```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data", enable_token_tracking=False)
```

Tracking failures never affect the answer — they are caught and logged. A custom provider
is tracked automatically if it exposes a `last_usage` attribute with `prompt_tokens` and
`completion_tokens`; one that does not is simply untracked.

## Not in 0.1.0

Several metrics are defined but never emitted, so they will not appear in your scrape:
model-version and prompt-version gauges, faithfulness and hallucination gauges, replay
counts, evaluation run counts, auth login counters, and the API request counters. They
belong to call sites that do not exist in this release.
