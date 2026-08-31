"""Decide which findings get verified, and by what.

Every surfaced finding gets the (free) deterministic pass. Only findings that
are still important or unresolved after that -- and within a hard cap -- go to
the LLM verifier.
"""

from __future__ import annotations

from pr_review.config import Config
from pr_review.models import Finding, Verdict, VerdictResult
from pr_review.verification.deterministic import EVIDENCE_NOT_ANCHORED

_SEVERITY_RANK = {"info": 0, "warning": 1, "blocker": 2}

# A finding that quotes evidence which is not in the file it cites is not
# anchored to anything -- the model paraphrased instead of quoting. That is a
# real negative signal, so it must not keep a near-certain confidence into the
# report. Capping into the uncertain band also routes it to LLM verification
# instead of letting it sail through.
UNANCHORED_CONFIDENCE_CAP = 0.6


class VerificationRouter:
    def __init__(self, config: Config) -> None:
        self.v = config.verification

    def adjust(self, finding: Finding, deterministic: VerdictResult) -> None:
        """Fold the deterministic outcome back into the finding, in place."""
        if EVIDENCE_NOT_ANCHORED in deterministic.checks:
            if finding.confidence > UNANCHORED_CONFIDENCE_CAP:
                finding.extra["confidence_capped_from"] = finding.confidence
                finding.confidence = UNANCHORED_CONFIDENCE_CAP
            finding.extra["evidence_anchored"] = False

    def needs_llm(self, finding: Finding, deterministic: VerdictResult) -> bool:
        if not self.v.enabled or not self.v.llm_enabled:
            return False
        if deterministic.verdict == Verdict.INVALID:
            return False  # already settled, and cheaply
        important = _SEVERITY_RANK.get(finding.severity, 0) >= _SEVERITY_RANK.get(
            self.v.llm_min_severity, 1
        )
        uncertain_conf = self.v.uncertain_low <= finding.confidence <= self.v.uncertain_high
        unresolved = deterministic.verdict != Verdict.VALID
        return unresolved and (important or uncertain_conf)

    def order(self, findings: list[Finding]) -> list[Finding]:
        return sorted(
            findings,
            key=lambda f: (
                -_SEVERITY_RANK.get(f.severity, 0),
                abs(f.confidence - 0.5),
            ),
        )
