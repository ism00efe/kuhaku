"""``BaseMetric``: the extension point every evaluation metric implements.

Adding a new metric never requires touching ``EvaluationRunner`` -- the runner only
knows how to call ``evaluate()`` and merge whatever it returns into the per-question
result dict. Scoring one ``EvaluationSample`` (not RAG-specific ``retrieved``/``golden``
objects) is what makes a metric class reusable for a non-RAG target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .sample import EvaluationSample


class BaseMetric(ABC):
    """Scores one ``EvaluationSample`` against the golden fields merged into it.

    ``evaluate()`` returns a dict of metric-name -> score so a single metric class can
    produce more than one score (e.g. a metric family parameterized by ``k``). An empty
    dict means "not applicable to this sample" (missing answer, empty context, judge
    unavailable, ...) and the runner skips recording it.
    """

    name: str

    @abstractmethod
    def evaluate(self, sample: EvaluationSample) -> dict[str, float]: ...
