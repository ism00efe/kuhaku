# Evaluation

`kuhaku.evaluation` measures retrieval and answer quality against a golden dataset. It is
tool-agnostic: it evaluates a `RAGEngine`, a bare retriever, or anything else that
implements the target contract.

Retrieval metrics need no language model. Answer-quality metrics need a judge you supply.

## The dataset

One JSON object per line — JSONL, not JSON.

```json
{"question_id": "q1", "question": "How do I reset my password?", "expected_sources": ["handbook"], "golden_answer": "Open Settings, then Security, then Reset password.", "category": "howto"}
{"question_id": "q2", "question": "When do deploys run?", "expected_sources": ["runbook"], "category": "factual"}
```

| Field | Required | Meaning |
|---|---|---|
| `question_id` | yes | your identifier for the row |
| `question` | yes | what gets asked |
| `expected_sources` | no | document ids that count as relevant — this is what retrieval metrics score against |
| `golden_answer` | no | reference answer, used by answer correctness |
| `category` | no | free-form label, carried into results |
| `golden_chunks` | no | loaded but not used by any metric in this release |

Blank lines are skipped. A missing optional field logs a warning and takes its default.
Only `expected_sources` drives the retrieval metrics — if it is empty, those metrics have
nothing to score.

## Running it

<!-- no-exec -->
```python
from kuhaku import RAG
from kuhaku.evaluation import (
    EvaluationRunner,
    HitRateAtKMetric,
    MRRMetric,
    NDCGAtKMetric,
    RecallAtKMetric,
)

rag = RAG(vector_store="./kuhaku-data")

runner = EvaluationRunner(
    [RecallAtKMetric(k=5), MRRMetric(), NDCGAtKMetric(k=5), HitRateAtKMetric(k=5)],
    "golden.jsonl",
)

summary = runner.run(rag.engine, top_k=5)
print(summary)      # {"recall_at_k": 0.82, "mrr": 0.71, ...}
```

`run()` returns the mean of each metric across the dataset. A row that raises is scored
as an empty sample rather than aborting the run, and a metric that raises is skipped for
that row — so a run always completes and you should check the summary for a metric that
went suspiciously quiet.

## Metrics

**Retrieval** — no model required, just `expected_sources`:

| Class | Result key | Argument |
|---|---|---|
| `RecallAtKMetric` | `recall_at_k` | `k=5` |
| `PrecisionAtKMetric` | `precision_at_k` | `k=5` |
| `HitRateAtKMetric` | `hit_rate_at_k` | `k=5` |
| `NDCGAtKMetric` | `ndcg_at_k` | `k=5` |
| `MRRMetric` | `mrr` | none |

**Answer quality:**

| Class | Result key | Needs |
|---|---|---|
| `AnswerCorrectnessMetric` | `answer_correctness` | `golden_answer` in the dataset |
| `FaithfulnessMetric` | `faithfulness` | a judge model |

The `k` on a metric is independent of the `top_k` you pass to `run()`. Retrieve 10 and
score recall at 5 if that is the question you are asking.

## The judge

`FaithfulnessMetric` scores whether the answer is supported by the retrieved context. It
needs a language model, supplied one of two ways.

Pass a provider directly:

<!-- no-exec -->
```python
from kuhaku.core.llm import build_llm_provider
from kuhaku import Settings
from kuhaku.evaluation import EvaluationRunner, FaithfulnessMetric

judge = build_llm_provider(Settings(llm_provider="openai", openai_api_key="sk-..."))

runner = EvaluationRunner(
    [FaithfulnessMetric()],
    "golden.jsonl",
    judge_llm_provider=judge,
)
```

Or let the runner build one from settings — it forces temperature to zero:

<!-- no-exec -->
```python
from kuhaku import Settings
from kuhaku.evaluation import EvaluationRunner, FaithfulnessMetric

runner = EvaluationRunner(
    [FaithfulnessMetric()],
    "golden.jsonl",
    settings=Settings(llm_provider="openai", openai_api_key="sk-..."),
)
```

**With neither, `faithfulness` silently does not appear in the summary.** The metric
returns nothing rather than failing. If you expected a faithfulness number and do not see
one, no judge was wired.

A metric constructed with its own judge always keeps it — the runner only fills in judges
that are missing.

## Storing results

Results go to an in-memory store by default. For runs you want to compare later:

<!-- no-exec -->
```python
from kuhaku.evaluation import EvaluationRunner, RecallAtKMetric, SqliteEvaluationStore

store = SqliteEvaluationStore("./eval-results.sqlite3")

runner = EvaluationRunner(
    [RecallAtKMetric(k=5)],
    "golden.jsonl",
    store=store,
    run_id="hybrid-baseline",
)
summary = runner.run(rag.engine, top_k=5)
store.close()
```

`run_id` defaults to a random hex string. Naming it is what makes two runs comparable.

`InMemoryEvaluationStore` additionally exposes `.results`, a list of per-row dictionaries
with the retrieved ids, the metrics and the answer — useful for finding *which* rows
failed rather than only the average.

## Evaluating something other than RAG

`EvaluationRunner.run()` accepts several shapes and adapts them:

- an object with `evaluate_sample(query) -> EvaluationSample` — used as-is
- an object with `ask(query)` or `answer(query)` — the result's `.text` and `.retrieved`
  are read
- an object with `retrieve(query, top_k)` or `search(query, top_k)`
- a plain callable

`RAGEngine` implements `evaluate_sample` directly, which is why `rag.engine` can be
passed straight in.

To write your own target, implement one method:

<!-- no-exec -->
```python
from kuhaku.evaluation import EvaluationSample

class MyTarget:
    def evaluate_sample(self, query: str) -> EvaluationSample:
        docs = my_search(query)
        return EvaluationSample(
            query=query,
            answer=my_answer(query, docs),
            contexts=[d.text for d in docs],
            retrieved_doc_ids=[d.id for d in docs],
        )
```

## Writing your own metric

```python
from kuhaku.evaluation import BaseMetric, EvaluationSample

class AnswerLengthMetric(BaseMetric):
    name = "answer_length"

    def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
        if not sample.answer:
            return {}
        return {"answer_length": float(len(sample.answer.split()))}
```

Return an empty dict when the sample lacks what you need — that is how the built-in
metrics opt out of a row without failing the run.
