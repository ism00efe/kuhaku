"""Prompt injection guard.

A deterministic, regex-based pre-filter that inspects the retrieval query — already
sanitized of PII by this point (see ``rag.engine.RAGEngine.answer``) — before it reaches
retrieval or the LLM. Mirrors ``sanitization.py``'s philosophy: rule-based, no LLM,
narrow and auditable. This is a coarse first line of defense against the well-known
override/jailbreak patterns, not general content moderation.

Known limitation: patterns target the common English jailbreak phrasing ("ignore
previous instructions", etc.). A determined attacker phrasing an injection in Turkish
could evade them, the same inherent limit deterministic regex sanitization already has.
"""

from __future__ import annotations

import logging
import random
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .normalizer import normalize

if TYPE_CHECKING:
    from .classifier import Stage1Result, Stage2Result, TwoStageClassifier

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = (
    "This request could not be processed for security reasons. Please rephrase "
    "your question."
)

# Attempts to override/replace the system prompt's instructions.
_INSTRUCTION_OVERRIDE = re.compile(
    r"(?i)\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?"
    r"(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|rules?)\b"
)

# Attempts to reassign the assistant's persona ("you are now...", classic jailbreaks).
_PERSONA_OVERRIDE = re.compile(
    r"(?i)\byou\s+are\s+now\b|\back\s+as\b.{0,20}\bDAN\b|\bjailbreak\b|\bDAN\s+mode\b"
)

# A line that tries to inject a new "system:" / "assistant:" / "user:" turn into the
# conversation, smuggling fake roles into what should be plain user content.
_ROLE_SWITCH = re.compile(r"(?im)^\s*(system|assistant|user)\s*:")

# Chat-template control tokens that have no business appearing in a user question.
_SPECIAL_TOKENS = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|\[/?INST\]|<<SYS>>|<</SYS>>|<\|endoftext\|>"
)

# Attempts to exfiltrate the system prompt itself.
_PROMPT_LEAK = re.compile(
    r"(?i)\b(reveal|print|show|display|repeat)\b.{0,30}\bsystem\s+prompt\b"
)

# Checked in order; first match wins. Order doesn't affect the outcome (any match
# blocks), only which `reason` is reported.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_INSTRUCTION_OVERRIDE, "instruction_override"),
    (_PERSONA_OVERRIDE, "persona_override"),
    (_ROLE_SWITCH, "role_switch"),
    (_SPECIAL_TOKENS, "special_token"),
    (_PROMPT_LEAK, "prompt_leak"),
)


def inspect_query(query: str) -> tuple[bool, str]:
    """Inspect ``query`` for prompt-injection patterns.

    Returns ``(safe, reason)``. ``safe=True`` means nothing suspicious was found and
    ``reason`` is ``"ok"``. ``safe=False`` means the query should be refused; ``reason``
    is a fixed, low-cardinality category name (never the raw match) safe to log or use
    as a metric label.
    """

    if not query:
        return True, "ok"
    for pattern, reason in _PATTERNS:
        if pattern.search(query):
            return False, reason
    return True, "ok"


# ---------------------------------------------------------------------------
# Prompt Injection Guard v2 — layered on top of the legacy filter above, which
# stays completely unchanged and keeps running first. Everything
# below is dormant unless `Settings.guard_enabled` is True (default False).
# ---------------------------------------------------------------------------

# Process-wide singleton, generated once at import. Logged once so it's checkable
# without reading source; the value itself is not sensitive (it exists
# specifically to be logged and to appear, if ever, in a leaked LLM response), so it's
# safe in a plain log message.
CANARY_TOKEN: str = uuid.uuid4().hex
logger.info("guard v2 canary token: %s", CANARY_TOKEN)

GUARD_REJECT_MESSAGE = (
    "This request could not be processed due to security policy. Please rephrase "
    "your question."
)
RESTRICTED_WARNING = (
    "⚠️ This response has been restricted to knowledge-base sources only, per "
    "security policy."
)


@dataclass(frozen=True)
class GuardDecision:
    """The outcome of one `GuardPipeline.evaluate()` call."""

    zone: str  # "pass" | "restricted" | "reject"
    stage1: Stage1Result
    stage2: Stage2Result
    escalation_reason: str | None  # None | "threshold" | "sampled" | "elevated_access"
    norm_drift: int
    normalized_length: int


def _decide_zone(
    stage1: Stage1Result,
    stage2: Stage2Result,
    *,
    low: float,
    high: float,
    stage2_confidence_floor: float = 0.7,
) -> str:
    """Three-zone decision.

    Precedence, most to least authoritative: a degraded Stage-2 forces RESTRICTED
    unconditionally — including over a high Stage-1 score. This is deliberate, not an
    oversight: without a deployed Stage-2 model (this change ships none), Stage-2
    is permanently degraded, so this is the accepted default posture of the whole v2
    pipeline until a real model exists — matching the `guard_enabled=False` rollout-gate
    decision. The legacy regex guard (`inspect_query`) still runs unconditionally before
    this and blocks the most blatant patterns outright, so a degraded Stage-2 does not
    mean *no* protection, only that v2's own REJECT zone is inert until Stage-2 exists.

    `stage2_confidence_floor` is hardcoded: no Settings field for it, since this branch
    is inert without a real Stage-2 model.
    """

    if stage2.degraded:
        return "restricted"
    if stage2.ran and stage2.label == "unsafe":
        return "reject"
    if stage1.score >= high:
        return "reject"
    if (
        stage2.ran
        and stage2.label == "safe"
        and (stage2.confidence or 0.0) < stage2_confidence_floor
    ):
        return "restricted"
    if stage1.score >= low:
        return "restricted"
    return "pass"


class GuardPipeline:
    """Orchestrates normalize -> classify (Stage1, optionally Stage2) -> 3-zone decide.

    kuhaku ships no composition root that builds this for you (see `Settings.guard_enabled`'s
    docstring): the caller constructs one -- when `guard_enabled` -- and injects it into
    `RAGEngine`, the same way `cache`/`retriever` already are (constructor kwarg, or
    `RAGEngine.update_guard()` after the fact).
    """

    def __init__(
        self,
        classifier: TwoStageClassifier,
        *,
        low_threshold: float,
        high_threshold: float,
        norm_drift_tolerance: int,
        sampling_rate: float,
        citation_grounding_threshold: float,
        guard_version: str,
        model_version: str,
        rng: random.Random | None = None,
    ) -> None:
        # Local import: avoids a module-level circular import (classifier.py imports
        # `guard._PATTERNS`), and only runs once per GuardPipeline construction, not
        # per-request.
        from .classifier import NOT_RUN_STAGE2

        self._not_run_stage2 = NOT_RUN_STAGE2
        self._classifier = classifier
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._norm_drift_tolerance = norm_drift_tolerance
        self._sampling_rate = sampling_rate
        self.citation_grounding_threshold = citation_grounding_threshold
        self.guard_version = guard_version
        self.model_version = model_version
        self._rng = rng or random.Random()

    def evaluate(self, query: str) -> GuardDecision:
        normalization = normalize(query)
        if normalization.drift > self._norm_drift_tolerance:
            logger.debug(
                "normalization drift exceeded tolerance",
                extra={"norm_drift": normalization.drift},
            )

        stage1 = self._classifier.classify_stage1(normalization.text)

        escalation_reason: str | None = None
        if stage1.score >= self.low_threshold:
            escalation_reason = "threshold"
        elif self._rng.random() < self._sampling_rate:
            escalation_reason = "sampled"

        stage2 = (
            self._classifier.classify_stage2(normalization.text)
            if escalation_reason
            else self._not_run_stage2
        )

        # Shadow evaluation (sampling-based escalation) is observability-only: Stage-2's
        # result is still returned on the decision for audit/drift-detection, but the zone
        # must be decided as if Stage-2 never ran, so a random 5% sample never changes
        # user-visible behavior.
        zone_stage2 = self._not_run_stage2 if escalation_reason == "sampled" else stage2
        zone = _decide_zone(stage1, zone_stage2, low=self.low_threshold, high=self.high_threshold)

        return GuardDecision(
            zone=zone,
            stage1=stage1,
            stage2=stage2,
            escalation_reason=escalation_reason,
            norm_drift=normalization.drift,
            normalized_length=normalization.normalized_length,
        )

    def validate(self) -> bool:
        """Startup self-check (``kuhaku.core.policy.enforce_security_policy``):
        confirms this pipeline can evaluate a sample query without raising. Returns
        ``True`` on success, raises the underlying exception otherwise.

        Deliberately not a check for "Stage-2 model present": Stage-1/Stage-2 already
        fail open by design (a missing model degrades rather than crashes, see
        ``TwoStageClassifier``/``Stage2Classifier``), so this only guards against
        ``GuardPipeline`` itself being unable to run at all.
        """

        self.evaluate("startup self-check")
        return True
