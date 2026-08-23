"""Cross-cutting security concerns beyond PII sanitization (see ``sanitization.py``)."""

from .audit import (
    read_audit_log,
)
from .audit import (
    read_recent as read_recent_audit_records,
)
from .audit import (
    record as record_audit,
)
from .classifier import (
    LRModelScorer,
    RuleBasedScorer,
    Stage1Result,
    Stage2Classifier,
    Stage2Result,
    TwoStageClassifier,
)
from .guard import (
    CANARY_TOKEN,
    GUARD_REJECT_MESSAGE,
    REFUSAL_MESSAGE,
    RESTRICTED_WARNING,
    GuardDecision,
    GuardPipeline,
    inspect_query,
)
from .normalizer import NormalizationResult, normalize
from .output_guard import SAFE_FALLBACK, OutputGuardResult, evaluate_output

__all__ = [
    "inspect_query",
    "REFUSAL_MESSAGE",
    # Prompt Injection Guard v2
    "CANARY_TOKEN",
    "GUARD_REJECT_MESSAGE",
    "RESTRICTED_WARNING",
    "GuardDecision",
    "GuardPipeline",
    "NormalizationResult",
    "normalize",
    "Stage1Result",
    "Stage2Result",
    "RuleBasedScorer",
    "LRModelScorer",
    "Stage2Classifier",
    "TwoStageClassifier",
    "OutputGuardResult",
    "SAFE_FALLBACK",
    "evaluate_output",
    "record_audit",
    "read_recent_audit_records",
    "read_audit_log",
]
