# Classification

`kuhaku.tools.classification` assigns labels from a fixed label set to a text: intent
detection, support triage, moderation-style labeling. It is a tool built on
`kuhaku.core` and imports nothing from any other tool.

Two rules shape it:

- **It abstains instead of guessing.** A text the classifier cannot settle comes back
  `UNDECIDED` with a machine-readable reason. It is never mapped to a default or
  "nearest" label. Where an unclassifiable text should go is your decision.
- **Every stage around the model is optional.** Deterministic rules can decide before
  the model is called, and your own checks can veto a decision afterwards. Each stage
  can be added, removed, replaced or switched off.

## A first classifier

The examples on this page pass a stand-in `llm` so they run offline. In real use, leave
`llm` out and the provider is built from your settings, the same way `RAG()` does it.
You can also pass any `LLMProvider`.

```python
import json

from kuhaku import Classifier, Label


class StubLLM:
    """Stands in for a real provider: always gives the same reply."""

    name = "stub:model"

    def generate(self, system: str, user: str) -> str:
        return json.dumps({
            "labels": [{"label": "technical", "confidence": 0.92}],
            "reason": "the app crashes",
        })


classifier = Classifier(
    [
        Label("billing", "Invoices, charges and refunds", examples=("Why was I charged twice?",)),
        Label("technical", "Bugs, errors, crashes"),
        Label("account", "Login, password, profile"),
    ],
    llm=StubLLM(),
)

result = classifier.classify("The app crashes when I open settings")
assert result.ok
assert result.label == "technical"
assert result.scores == {"technical": 0.92}
assert result.decided_by == "classify"
print(result.reason, result.trace_id)
```

A label is a name plus an optional description and examples. All three are shown to the
model, and only names from the label set are accepted back. Bare strings work too
(`Classifier(["spam", "ham"], ...)`), but then the model has only the name to go on.

## The result

`ClassificationResult` has these fields:

| field | meaning |
|---|---|
| `status` | `DECIDED` or `UNDECIDED` |
| `labels` | the chosen labels, highest confidence first; empty when undecided |
| `label` | the first of `labels`, or `None` |
| `scores` | the confidence of every label the deciding stage proposed, including those below the threshold; a rule match scores `1.0` |
| `decided_by` | the stage that produced the outcome: `"rules"`, `"classify"`, `"validate"`, or your own step's name |
| `reason` | why, in words: the rule that matched, the model's rationale, or why it abstained |
| `undecided` | an `UndecidedReason` when undecided, else `None` |
| `attempts` | how many model calls were made, counting repairs and fallbacks |
| `usage` | the summed `TokenUsage` of those calls, or `None` when no call was made or the provider reports no usage |
| `metadata` | `llm` (the provider whose answer was used), `rationale`, `rule_conflict` |
| `trace_id` | the id every log record and span of this call carries |

`UndecidedReason` has these values:

- `NO_LABEL`: the model said no label fits.
- `LOW_CONFIDENCE`: no proposed label reached `min_confidence` (default `0.5`).
- `AMBIGUOUS`: in single-label mode, two labels tied for the top confidence.
- `UNKNOWN_LABEL`: the model kept naming labels outside the label set, even after
  repairs.
- `UNPARSEABLE`: the reply was not usable JSON, even after repairs.
- `REJECTED`: a decision was vetoed by a validator.

If the provider cannot be reached at all, `classify` raises `LLMUnavailableError` and
returns no result. An unreachable service says nothing about the text, so it is not
reported as an outcome.

## Single-label and multi-label

By default the classifier picks exactly one label or none. It takes the top label when
that label reaches `min_confidence` and is not tied. With `multi_label=True` it takes
every label that reaches `min_confidence`:

```python
import json

from kuhaku import Classifier


class StubLLM:
    name = "stub:model"

    def generate(self, system: str, user: str) -> str:
        return json.dumps({"labels": [
            {"label": "technical", "confidence": 0.9},
            {"label": "account", "confidence": 0.7},
            {"label": "billing", "confidence": 0.1},
        ]})


classifier = Classifier(["billing", "technical", "account"], multi_label=True, llm=StubLLM())
assert classifier.classify("I can't log in and the app crashes").labels == ("technical", "account")
```

## Rules before the model

A `KeywordRule`, a `RegexRule`, or any callable `ctx -> label | labels | None` can decide
without a model call. Rules are passed as `rules=`, and they run in a `rules` stage
placed before the model:

```python
from kuhaku import Classifier
from kuhaku.tools.classification import KeywordRule, RegexRule


class NeverCalled:
    name = "stub:model"

    def generate(self, system: str, user: str) -> str:
        raise AssertionError("a rule should have decided")


classifier = Classifier(
    ["billing", "technical", "account"],
    llm=NeverCalled(),
    rules=[
        KeywordRule("billing", ["invoice", "refund"]),
        RegexRule("technical", r"\berror \d{3}\b"),
    ],
)

result = classifier.classify("Where is my refund?")
assert result.labels == ("billing",)
assert result.decided_by == "rules"
assert result.attempts == 0
```

Keyword rules are case-insensitive and match whole words by default (`case_sensitive=`,
`whole_word=`). In single-label mode, a rule decides only when every rule that fired
names the same label. If they disagree, the disagreement is recorded in
`metadata["rule_conflict"]` and the model decides, so the order you list rules in never
becomes a hidden priority. In multi-label mode, the labels of all fired rules together
are the decision and the model is not called. Use rules there only for texts whose
labels they fully determine.

A rule that names a label outside the label set raises `ConfigError` on the first
`classify` call.

## Validators after the decision

A `ValidationStep` runs your checks on any decision, whichever stage made it. A
validator returns `None` to accept the decision, or a reason string to veto it:

```python
import json
import re

from kuhaku import Classifier, UndecidedReason
from kuhaku.tools.classification import ValidationStep


class StubLLM:
    name = "stub:model"

    def generate(self, system: str, user: str) -> str:
        return json.dumps({"labels": [{"label": "billing", "confidence": 0.9}]})


def billing_needs_an_order_number(labels, ctx):
    if "billing" in labels and not re.search(r"#\d+", ctx.text):
        return "billing requests need an order number"
    return None


classifier = Classifier(
    ["billing", "technical"],
    llm=StubLLM(),
    steps=[ValidationStep(billing_needs_an_order_number)],
)

result = classifier.classify("refund please")
assert result.undecided is UndecidedReason.REJECTED
assert result.decided_by == "validate"
assert classifier.classify("refund for order #42").label == "billing"
```

A vetoed result becomes `UNDECIDED`. The validator never swaps in a different label,
because what a vetoed text should get instead is not something it can know.

## Repairs, escalation and context

- **Repairs.** If a reply is not usable JSON, or it names labels outside the label set,
  it is sent back to the model with the problems listed. This happens up to
  `max_repair_attempts` times (default 1). Unknown names are never dropped quietly.
- **Escalation.** `fallback_llms=[stronger]` tries further providers, in order, whenever
  the one before did not reach a decision: it was unreachable, its output was unusable,
  or it abstained. A cheap model handles the clear cases, and the stronger one only
  sees the rest. If a fallback is unreachable, an earlier provider's abstention is kept
  rather than raising.
- **Conversation context.** `classify(text, history=[Message(...), ...])` shows earlier
  turns to the model so it can work out what `text` refers to. "Yes, cancel it" is only
  a cancellation in context.
- **Instructions.** `Classifier(..., instructions="...")` adds your domain rules to the
  system prompt. They go after the framework's own rules, which they cannot replace.
  The framework's rules treat the text as data, not instructions, and fence it with a
  random boundary on every call.

## Composing the pipeline

`classifier.pipeline` works the same way as the extraction pipeline. You can use
`add(step, before=/after=)`, `remove`, `replace`, `enable` and `disable`. A step
subclasses `Step` and has two hooks. `prepare(ctx)` can contribute a prompt rule
(`ctx.instructions`) or a titled input block (`ctx.prompt_blocks`). `run(ctx)` reads the
context and records an outcome with `ctx.decide(...)` or `ctx.abstain(...)`.

A step that calls a model should do it through `ctx.generate(llm, system, user)`, so
the call is counted in `attempts` and `usage`. The `classify` step cannot be removed or
disabled, because it guarantees that every text ends decided or undecided. It can be
replaced by another step with the same name.

## Observability

Each `classify` call runs under `bind_trace_id`. It opens a `classification` span and
log record that carry the outcome, `decided_by`, `undecided` and `attempts`, plus one
`classification.<step>` span per stage. Every provider is wrapped in `TokenTrackingLLM`
unless you pass `enable_token_tracking=False`, so token metrics and GenAI span
attributes are recorded the same way as for `RAG`. Retries with backoff for transient
failures happen inside the providers themselves.

## Evaluation

`Classifier` implements `EvaluationTarget`, so you can pass it straight to
`EvaluationRunner`. Each sample's `answer` is the chosen labels joined by spaces, or
`None` when undecided. Against a golden answer made of label names,
`AnswerCorrectnessMetric` therefore gives exact-match accuracy for single-label
classification and Jaccard overlap of label sets for multi-label classification. This
only holds when label names contain no spaces. The full outcome is in
`sample.metadata`. There is no dedicated label metric yet: no per-label precision or
recall, and no confusion matrix.

## Not in this tool yet

- **No security steps.** PII masking and the prompt-injection input guard are not wired
  into the classification pipeline, because how core security attaches to tools other
  than RAG has not been settled. The prompt does treat the text as data and fences it,
  but that is not the input guard.
- **No batch API and no async.**
- **No calibrated confidences.** Confidences are what the model reports. Choose
  `min_confidence` by measuring on your own data, not by its face value.
