"""Text classification: a text -> labels from a fixed label set, as a kuhaku tool.

Intent detection, triage, moderation-style labeling. Built only on ``kuhaku.core`` (the
LLM abstraction, observability, shared models) and ``kuhaku.evaluation``'s sample type;
it imports nothing from another tool.

    from kuhaku.tools.classification import Classifier, KeywordRule, Label

    classifier = Classifier(
        [Label("billing", "Invoices, charges, refunds"), Label("technical", "Bugs, errors")],
        rules=[KeywordRule("billing", ["invoice", "refund"])],
    )
    result = classifier.classify("The app crashes when I log in")
    result.status, result.label, result.scores, result.decided_by, result.reason
"""

from .classifier import Classifier
from .context import ClassificationContext
from .labels import Label, LabelSet
from .parsing import OutputParseError
from .pipeline import CLASSIFY_STEP, Pipeline, Step
from .result import ClassificationResult, ClassificationStatus, UndecidedReason
from .steps import (
    RULES_STEP,
    ClassifyStep,
    KeywordRule,
    RegexRule,
    Rule,
    RuleStep,
    ValidationStep,
    Validator,
)

__all__ = [
    "CLASSIFY_STEP",
    "RULES_STEP",
    "ClassificationContext",
    "ClassificationResult",
    "ClassificationStatus",
    "Classifier",
    "ClassifyStep",
    "KeywordRule",
    "Label",
    "LabelSet",
    "OutputParseError",
    "Pipeline",
    "RegexRule",
    "Rule",
    "RuleStep",
    "Step",
    "UndecidedReason",
    "ValidationStep",
    "Validator",
]
