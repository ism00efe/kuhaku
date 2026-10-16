# Structured extraction

Structured extraction turns free text into a validated Pydantic model: an order out of a
chat message, an appointment out of an email, an invoice out of a pasted document. It is
kuhaku's second tool, built on `kuhaku.core` the same way RAG is, and it lives in
`kuhaku.tools.extraction`.

You describe what to extract; kuhaku asks the language model, checks the reply against
your description, asks again when the reply is malformed, and tells you plainly whether
the text contained everything you asked for.

## Your first extraction

```python
from pydantic import BaseModel
from kuhaku import Extractor


class Appointment(BaseModel):
    """A message asking for an appointment."""

    department: str
    day: str
    time: str


result = Extractor(Appointment).extract("I'd like a cardiology appointment tomorrow")

print(result.status)         # ExtractionStatus.NEEDS_CLARIFICATION
print(result.missing_slots)  # ('/time',)
print(result.data)           # {'department': 'cardiology', 'day': 'tomorrow', 'time': None}
```

The model is chosen the same way as for RAG — see [Providers](providers.md). The class
is the single source of truth: the JSON Schema the model sees is generated from it, and
the reply is validated against it.

## Without writing a class

To try something quickly, describe the fields instead:

```python
from kuhaku import Extractor

# Every field a required string.
contact = Extractor(["name", "email"]).extract("I'm Ada, ada@example.com")

# A type per field; a type that accepts None makes the field optional.
person = Extractor({"name": str, "age": int, "note": str | None}).extract("Ada, 36")

# (type, description) tells the model what a field means.
# Nested objects and lists of them are written the same way.
invoice = Extractor(
    {
        "lines": [{"item": str, "price": float}],
        "total": (float, "grand total including VAT"),
    },
    name="Invoice",
    description="A supplier invoice.",
).extract("2 x tea at 3.50, total 7.00")

print(contact.status, person.status, invoice.status)
```

`name` and `description` play the role of a class name and docstring. Add `instructions`
for domain rules the model should follow, such as "copy product names exactly as
written".

## Reading the result

`extract()` returns an `ExtractionResult`. Its `status` says what happened, without you
having to inspect the extracted values:

| Status | Meaning |
|---|---|
| `COMPLETE` | Every required field was extracted and every enabled check passed |
| `NEEDS_CLARIFICATION` | Required values are not in the text — ask the user for them |
| `INVALID` | Values were extracted, but a check (evidence, your rules) rejected them |
| `FAILED` | The model gave no usable output, even after the allowed retries |
| `NO_MATCH` | A router found no schema that fits the text |
| `BLOCKED` | The input guard refused the input; no model saw it |

The other fields:

- `value` — an instance of your model, set only when the status is `COMPLETE`.
- `data` — the model's last output as a plain dict, kept even when it did not validate,
  so you can show what was understood so far.
- `missing_slots` — JSON Pointers of the required fields the text did not provide.
- `issues` — every problem found, each with a `kind`, a `code`, a `message` and the
  `path` of the field concerned.
- `fields` — per-field evidence reports, filled only by the evidence step.
- `attempts` — how many model calls the extraction took.
- `metadata["llm"]` — which provider's reply the result came from.
- `trace_id` — ties the extraction's log lines and spans together.

A value the text does not contain is never asked for again: a retry cannot make it
appear. It is reported as missing instead.

## Asking for what is missing

Conversations rarely deliver every value at once. Pass the previous result back as
`prior=` and the new message fills the gaps — this is **slot filling**:

```python
from pydantic import BaseModel
from kuhaku import Extractor


class Appointment(BaseModel):
    department: str
    day: str
    time: str


extractor = Extractor(Appointment)

first = extractor.extract("A cardiology appointment tomorrow, please")
# first.status is NEEDS_CLARIFICATION, first.missing_slots is ('/time',)

second = extractor.extract("3 pm", prior=first)
print(second.status, second.data)
```

A value from this message wins; a field this message leaves out keeps its earlier
value. kuhaku keeps no state between calls — you hand the previous result back, the same
way you hand back the conversation.

Two more inputs help the model resolve what a message refers to without being extracted
from themselves:

```python
from kuhaku import Extractor, Message

result = Extractor({"product": str, "quantity": float}).extract(
    "make that two",
    history=[Message("user", "one box of tea please")],
    reference={"cart": [{"product": "tea", "quantity": 1}]},
)
print(result.status)
```

`history` is earlier conversation turns; `reference` is any data you already hold, such
as the current cart. Both are shown to the model as data, never as instructions.

## Routing between schemas

When a message could be one of several things, put an extractor per schema behind an
`ExtractionRouter`. It first decides which schema the text belongs to, then extracts with
that schema's extractor:

```python
from pydantic import BaseModel
from kuhaku import ExtractionRouter, Extractor


class Appointment(BaseModel):
    """A message asking for a new appointment."""

    department: str
    day: str


class Cancellation(BaseModel):
    """A message cancelling an existing appointment."""

    department: str
    day: str


router = ExtractionRouter([Extractor(Appointment), Extractor(Cancellation)])

result = router.extract("Please cancel tomorrow's cardiology appointment")
print(result.schema_name, result.status)
```

The router chooses by each schema's name and description — the class docstring, or
`description=` — so give every schema one. Choosing a schema is itself an extraction of
a single field, so it gets the same retries, fallback models and security as any other.

When no schema fits, the result is `NO_MATCH`. The router never falls back to a default
schema: where an unclassifiable message should go is your decision.

Routes are a plain list, tried in order; the first that names a schema wins. A cheap
keyword rule in front of the model saves a model call when the answer is obvious:

```python
from pydantic import BaseModel
from kuhaku import ExtractionRouter, Extractor


class Appointment(BaseModel):
    """A message asking for a new appointment."""

    department: str


class Cancellation(BaseModel):
    """A message cancelling an existing appointment."""

    department: str


def cancel_keyword(request):
    return "Cancellation" if "cancel" in request.text.lower() else None


router = ExtractionRouter([Extractor(Appointment), Extractor(Cancellation)])
router.routes.insert(0, cancel_keyword)

print(router.route("cancel my cardiology appointment"))  # Cancellation
```

A route returns a schema name, or `None` to let the next route decide. Your own routing
rules for the model go in `LLMRoute(instructions="...")`, passed as
`ExtractionRouter(..., routes=[LLMRoute(instructions=...)])`.

`prior=` works through a router too; it is carried over only when the new message routes
to the same schema, so a follow-up that changes topic starts fresh.

## Steps

An extraction runs as a pipeline of named steps, in order:

| Step | What it does | On by default |
|---|---|---|
| `security` | core's input checks — see [Security](#security) | yes, every check off |
| `extract` | asks the model, parses and validates the reply, retries | yes, required |
| `slot_filling` | merges the reply into `prior=` | yes, inert without `prior=` |
| `evidence` | checks each value against a quote from the text | no |
| `validate` | runs your own rules | no |

Optional steps are passed as `steps=` and run after the defaults:

```python
from pydantic import BaseModel
from kuhaku.tools.extraction import (
    EvidenceStep,
    Extractor,
    Issue,
    IssueKind,
    ValidationStep,
)


class Order(BaseModel):
    product: str
    quantity: float


def at_most_100(order, ctx):
    if order.quantity > 100:
        return [Issue(IssueKind.INVALID, "too_many", "at most 100 per order", "/quantity")]
    return []


extractor = Extractor(
    Order,
    steps=[EvidenceStep(on_ungrounded="invalid"), ValidationStep(at_most_100)],
)
result = extractor.extract("5 boxes of tea")
print(result.status, result.fields)
```

**Evidence.** The model quotes the passage each value came from, and kuhaku checks the
quote against the text — a plain substring check, not another model call. Each field is
reported in `result.fields` as `SOURCE` (quoted and found), `PRIOR` (carried over from
`prior=`), `MISMATCH` (the quote is real but does not state the value), `UNGROUNDED` (no
quote, or a quote not in the text) or `MISSING`. With `on_ungrounded="invalid"` anything
not grounded makes the result `INVALID`; the default only reports it. A value written in a
form the text does not use — a date for "tomorrow" — is `MISMATCH`.

**Your rules.** A rule receives the validated model instance and the extraction context,
and returns issues: `IssueKind.INVALID` for a value that breaks a rule,
`IssueKind.NEEDS_CLARIFICATION` for something only the user can supply.

The pipeline can be rearranged after construction:

```python
from kuhaku.tools.extraction import EvidenceStep, Extractor, Step


class LowercaseText(Step):
    name = "lowercase"

    def prepare(self, ctx):
        ctx.text = ctx.text.lower()


extractor = Extractor({"product": str})
extractor.pipeline.add(LowercaseText(), before="extract")
extractor.pipeline.add(EvidenceStep())
extractor.pipeline.disable("evidence")
print(extractor.pipeline)
```

`add(step, before=/after=)`, `remove(name)`, `replace(name, step)`, `enable(name)` and
`disable(name)`. The `extract` step can be replaced but not removed or disabled —
without it there is nothing to check.

A step has two hooks. `prepare(ctx)` runs for every step before any step runs, which is
where a step changes what the model will see. `run(ctx)` runs in order afterwards. Steps
talk to each other only through `ctx`, so any of them can be swapped without touching the
rest.

## Retries and fallback models

A reply that is not valid JSON, or does not match the schema, is sent back to the model
with the problem described, up to `max_retries` more times (default 1).

`fallback_llms` are tried in order when a provider cannot produce usable output or cannot
be reached — a cheap model first, a stronger one only for the texts it gets wrong:

<!-- no-exec -->
```python
from kuhaku import Extractor, Settings, build_llm_provider

cheap = build_llm_provider(Settings(llm_provider="groq"))
strong = build_llm_provider(Settings(llm_provider="openrouter"))

extractor = Extractor({"product": str}, llm=cheap, fallback_llms=[strong], max_retries=2)
```

Each provider gets its own retries. A missing value never moves to the next model — a
stronger model cannot find what the text does not say. The result is built from exactly
one provider's reply, named in `result.metadata["llm"]`. `LLMUnavailableError` is
raised only when no provider replied at all.

## Prompts

Every text kuhaku sends can be read before it is sent, and almost all of it replaced:

```python
from kuhaku import Extractor

extractor = Extractor(
    {"product": str, "quantity": float},
    prompts={"rules": "You read wholesale orders sent over chat."},
)
system, user = extractor.preview("5 boxes of tea")
print(system)
print(user)
```

`preview()` builds the exact first request without calling the model. The replaceable
texts are the fields of `ExtractionPrompts`: `rules`, `retry`, `evidence`,
`slot_filling` and `route`. The rules that the text is data and not instructions, and
the reply format the parser depends on, are `FIXED_RULES` and cannot be replaced.

## Settings

Every extractor shares one set of settings, read from `KUHAKU_EXTRACTION__` environment
variables. Set something once and every extraction — the router's own included — follows
it:

| Setting | Variable | Default |
|---|---|---|
| `max_retries` | `KUHAKU_EXTRACTION__MAX_RETRIES` | `1` |
| `audit_log_path` | `KUHAKU_EXTRACTION__AUDIT_LOG_PATH` | unset — no audit log |
| `security.sanitize` | `KUHAKU_EXTRACTION__SECURITY__SANITIZE` | `false` |
| `security.input_guard` | `KUHAKU_EXTRACTION__SECURITY__INPUT_GUARD` | `false` |

The same settings can be built in code. One extractor that needs something different
gets its own; an argument passed to `Extractor` wins over both:

```python
from kuhaku.core.security import InputSecurity
from kuhaku.tools.extraction import ExtractionRouter, ExtractionSettings, Extractor

shared = ExtractionSettings(max_retries=2, security=InputSecurity(input_guard=True))

orders = Extractor({"product": str}, name="Order", extraction_settings=shared)
complaints = Extractor({"topic": str}, name="Complaint", extraction_settings=shared)
router = ExtractionRouter([orders, complaints])  # uses the first extractor's settings

one_off = Extractor({"product": str}, extraction_settings=shared, max_retries=0)
print(router.extraction_settings.security, one_off.pipeline)
```

These settings belong to the extraction tool, not to core's `Settings` — kuhaku's core
never depends on a tool.

## Security

Two protections are always on and cost nothing. Everything from outside the framework —
the text, the history, the reference, earlier values — is wrapped in a boundary that is
random per request, so text inside it cannot pose as instructions. And the rules that it
is data, not instructions, are part of `FIXED_RULES`, which no prompt replacement
removes.

Two more are available and off by default, because each gets in the way of ordinary
use:

- **PII masking** (`sanitize`) — e-mails, tokens, IP addresses, card and national ID
  numbers and phone numbers are replaced with placeholders such as `[EMAIL]` before any
  model sees them. A schema that asks for an e-mail or a phone number then gets the
  placeholder. What was masked — labels and counts, never values — is in
  `result.metadata["redactions"]`.
- **Input guard** (`input_guard`) — when the text, the history, the reference or the
  earlier turn's values match a known prompt-injection pattern, the extraction is
  refused with status `BLOCKED`, and no model is called. It also refuses chat logs with
  `user:` lines and texts that only quote such a phrase, and its patterns are English.

Both come from core's `InputPolicy`, the same checks every kuhaku tool can use; turn them
on through the settings above. With the guard on, a router refuses the text before its
own model call too.

## Audit log

Off by default, because it writes a file into your project. Give it a path:

```python
import tempfile
from pathlib import Path

from kuhaku import Extractor

log = Path(tempfile.mkdtemp()) / "extraction_audit.jsonl"
Extractor({"product": str}, audit_log_path=str(log)).extract("5 boxes of tea")
print(log.read_text(encoding="utf-8"))
```

One JSON line per `extract()` call, whatever the outcome — including a call that raised.
A record holds the trace id, a hash of the text and a PII-masked excerpt, the schema, the
status, the issue codes and the provider; never the raw text. A router given
`audit_log_path` also records the outcomes it ends itself (`BLOCKED`, `NO_MATCH`).

## Observability

Extraction emits the same logs, spans and metrics as the rest of kuhaku — see
[Observability](observability.md). The spans are `extraction` for the call,
`extraction.<step>` for each step, and `extraction.router` and `extraction.route` for
routing; their durations land on `kuhaku_stage_duration_seconds`. Token counts are
recorded for every provider, fallbacks included.

Pass your own request id to tie an extraction to the rest of your application's
records:

```python
from kuhaku import Extractor

result = Extractor({"product": str}).extract("5 boxes of tea", trace_id="req-1234")
print(result.trace_id)  # req-1234
```

## Measuring it

An `Extractor` and an `ExtractionRouter` are evaluation targets. Score them with
`StructuredMatchMetric` against a golden dataset whose `golden_answer` is the expected
object as JSON:

```python
import json
import tempfile
from pathlib import Path

from kuhaku import Extractor
from kuhaku.evaluation import EvaluationRunner, StructuredMatchMetric

dataset = Path(tempfile.mkdtemp()) / "golden.jsonl"
rows = [
    {"question_id": "1", "question": "5 boxes of tea",
     "golden_answer": json.dumps({"product": "tea", "quantity": 5})},
]
dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

extractor = Extractor({"product": str, "quantity": float})
summary = EvaluationRunner([StructuredMatchMetric()], dataset).run(extractor)
print(summary)  # {"field_accuracy": ..., "exact_match": ...}
```

`field_accuracy` is the share of the expected values the extraction got right;
`exact_match` is 1 only when all of them are right and no list has extra items. Strings are compared ignoring case
and surrounding spaces, numbers by value. A router answers with
`{"schema": ..., "data": ...}`, so one golden answer checks the routing and the values
together. See [Evaluation](evaluation.md) for the runner and the dataset format.

## Limits

- Because a missing value in a new message means "keep the earlier one", slot filling
  cannot clear a value back to empty.
- Evidence is a text match. A value the schema stores in a different form than the text
  writes it — a date for "tomorrow", a number for "a dozen" — is reported as `MISMATCH`.
- The input guard's patterns are English.
- There is no async or streaming API.
