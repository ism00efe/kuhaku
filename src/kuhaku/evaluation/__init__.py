"""Reusable, tool-agnostic evaluation infrastructure: a golden dataset loader, an
``EvaluationSample``/``EvaluationTarget`` contract, a pluggable metric abstraction, and a
central runner.

Lives at ``kuhaku.evaluation`` (not under any tool's own namespace) because it has no
dependency on RAG or any other specific tool -- ``TargetAdapter`` speaks only in terms of
generic shapes (``evaluate_sample()``, ``ask()``/``answer()``, ``retrieve()``/``search()``,
plain callables). ``kuhaku.tools.rag.engine.RAGEngine`` implements ``EvaluationTarget``
directly (see its own ``evaluate_sample``); a bare retriever or any other target goes
through ``TargetAdapter`` -- or a tool-specific subclass of it, e.g.
``kuhaku.tools.rag.evaluation_target.RAGTargetAdapter``, for tool-flavored answer
generation.
"""

from .base_metric import BaseMetric
from .dataset import EvaluationItem, load_eval_dataset
from .judges import LLMFaithfulnessEvaluator
from .metrics import (
    AnswerCorrectnessMetric,
    FaithfulnessMetric,
    HitRateAtKMetric,
    MRRMetric,
    NDCGAtKMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
    answer_correctness,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .runner import EvaluationRunner
from .sample import EvaluationSample
from .store import EvaluationStore, InMemoryEvaluationStore, SqliteEvaluationStore
from .target import EvaluationTarget, TargetAdapter

__all__ = [
    "BaseMetric",
    "EvaluationItem",
    "load_eval_dataset",
    "LLMFaithfulnessEvaluator",
    "AnswerCorrectnessMetric",
    "FaithfulnessMetric",
    "HitRateAtKMetric",
    "MRRMetric",
    "NDCGAtKMetric",
    "PrecisionAtKMetric",
    "RecallAtKMetric",
    "answer_correctness",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "EvaluationStore",
    "InMemoryEvaluationStore",
    "SqliteEvaluationStore",
    "EvaluationRunner",
    "EvaluationSample",
    "EvaluationTarget",
    "TargetAdapter",
]
