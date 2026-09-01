"""What it takes to call something a blocker.

"Blocker" is the only severity wired to automation: `fail-on-blocker` turns it
into a red build. The first real run produced
`[BLOCKER] CHANGELOG.md · conf 0.50 · unverified` -- a claim about code,
anchored to a changelog line, that the model itself was half sure of. Asserting
the label has to cost something, so a blocker must survive verification or be
confident enough to sit outside the uncertain band. Both bars already existed;
neither was applied to severity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pr_review.config import Config, VerificationConfig
from pr_review.models import (
    DeterministicAnalysis,
    Finding,
    PRContext,
    RepoMetadata,
    Verdict,
    VerdictResult,
    VerifiedFinding,
)
from pr_review.verification.deterministic import EVIDENCE_NOT_ANCHORED, DeterministicVerifier
from pr_review.verification.router import VerificationRouter


def _finding(**kw) -> Finding:
    base = dict(
        axis="correctness", severity="blocker", file="src/thing.py", line=None,
        issue="guard is documented but never wired in", evidence="quoted line",
        confidence=0.5,
    )
    base.update(kw)
    return Finding(**base)


def _router(**kw) -> VerificationRouter:
    return VerificationRouter(Config(verification=VerificationConfig(**kw)))


@pytest.mark.parametrize(
    "verdict, confidence, expected",
    [
        # neither verified nor confident -> a warning for a human to weigh
        (Verdict.UNCERTAIN, 0.50, "warning"),
        (Verdict.UNCERTAIN, 0.79, "warning"),
        # verification carries it, however hedged the model was
        (Verdict.VALID, 0.50, "blocker"),
        # confidence outside the uncertain band carries it on its own
        (Verdict.UNCERTAIN, 0.80, "blocker"),
        (Verdict.UNCERTAIN, 0.95, "blocker"),
    ],
)
def test_a_blocker_must_be_verified_or_confident(verdict, confidence, expected):
    vf = VerifiedFinding(_finding(confidence=confidence), VerdictResult(verdict))
    _router().enforce_blocker_bar(vf)
    assert vf.finding.severity == expected


def test_a_demotion_says_why():
    vf = VerifiedFinding(_finding(confidence=0.5), VerdictResult(Verdict.UNCERTAIN))
    _router().enforce_blocker_bar(vf)
    assert vf.finding.extra["severity_capped_from"] == "blocker"
    assert "0.50" in vf.finding.extra["severity_capped_reason"]


def test_lesser_severities_are_untouched():
    for sev in ("warning", "info"):
        vf = VerifiedFinding(
            _finding(severity=sev, confidence=0.1), VerdictResult(Verdict.UNCERTAIN)
        )
        _router().enforce_blocker_bar(vf)
        assert vf.finding.severity == sev


def test_the_bar_can_be_switched_off():
    vf = VerifiedFinding(_finding(confidence=0.1), VerdictResult(Verdict.UNCERTAIN))
    _router(blocker_requires_evidence=False).enforce_blocker_bar(vf)
    assert vf.finding.severity == "blocker"


def test_the_original_bad_blocker_cannot_recur(tmp_path: Path):
    """The exact shape that failed: a narrative 'quote' on a changelog."""
    (tmp_path / "CHANGELOG.md").write_text("## 0.2.0\n- guard_single_writer added\n")
    finding = _finding(
        file="CHANGELOG.md",
        evidence="The guard is documented as existing but is not wired into any "
        "store's write path.",
        confidence=0.99,
    )
    pr = PRContext("t", "", "main", "f", "0", "1", "")

    det = DeterministicVerifier().verify(
        finding, repo_root=tmp_path, pr=pr,
        analysis=DeterministicAnalysis(), meta=RepoMetadata(),
    )
    assert EVIDENCE_NOT_ANCHORED in det.checks

    router = _router()
    router.adjust(finding, det)          # unquotable -> confidence capped to 0.6
    vf = VerifiedFinding(finding, det)
    router.enforce_blocker_bar(vf)       # ... and 0.6 cannot carry a blocker

    assert finding.confidence == 0.6
    assert finding.severity == "warning"
