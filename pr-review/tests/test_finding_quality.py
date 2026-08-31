"""Guards against the failure that matters most: a confident, wrong blocker.

A real run produced `[BLOCKER] .gitignore · scope · verified · conf 0.99` for a
documentation-only PR. Three separate weaknesses lined up to make that possible:
the scope axis could emit `blocker`; the finding cited an arbitrary file because
it had to cite one; and the deterministic pass noticed that the "quoted"
evidence appears nowhere in that file, then dropped the observation on the floor
instead of passing it to the LLM verifier. Each is pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pr_review.axes.base import AXES, allowed_severities, clamp_severity
from pr_review.config import Candidate, Config, ProviderConfig, TierConfig
from pr_review.models import (
    ContextSpec,
    DeterministicAnalysis,
    FileChange,
    Finding,
    PRContext,
    ReviewContext,
    ReviewTask,
    Verdict,
    VerdictResult,
)
from pr_review.providers.dispatch import Completion
from pr_review.providers.selector import ModelSelector
from pr_review.review.reviewer import Reviewer
from pr_review.source.local_git import LocalGitSource
from pr_review.verification.deterministic import (
    EVIDENCE_NOT_ANCHORED,
    NO_FILE_CITED,
    DeterministicVerifier,
)
from pr_review.verification.llm import LLMVerifier
from pr_review.verification.router import UNANCHORED_CONFIDENCE_CAP, VerificationRouter


class _StubDispatcher:
    """Returns canned model output and records the prompts it was given."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, tier, prompt, *, system=None, temperature=None):
        self.prompts.append(prompt)
        return Completion(self.text, Candidate("stub", "stub-model", 512, 0.2, tier))


def _config() -> Config:
    return Config(
        providers={"stub": ProviderConfig(kind="mock", default_model="stub-model")},
        tiers={
            "basic": TierConfig("stub", "stub-model"),
            "verify": TierConfig("stub", "stub-model"),
        },
        fallback_provider="",
    )


def _task(axis: str) -> ReviewTask:
    return ReviewTask(
        axis=axis, depth="basic", model_tier="basic",
        context_strategy="default", context_spec=ContextSpec(),
    )


def _pr() -> PRContext:
    return PRContext(
        title="feat: add capability resolver",
        body="Adds the resolver and wires it into the provider.",
        base_ref="main", head_ref="feature", base_sha="a", head_sha="b",
        diff="", changed_paths=(".gitignore", "docs/configuration.md"),
    )


def _analysis() -> DeterministicAnalysis:
    a = DeterministicAnalysis()
    a.files = [
        FileChange(path=".gitignore", status="modified", additions=2, deletions=0),
        FileChange(path="docs/configuration.md", status="modified", additions=40, deletions=3),
    ]
    return a


# --------------------------------------------------------------------------- #
# 1. A process axis cannot fail a build
# --------------------------------------------------------------------------- #


def test_scope_and_method_cannot_emit_blocker():
    for name in ("scope", "method"):
        axis = AXES.create(name)
        assert "blocker" not in allowed_severities(axis)
        assert clamp_severity(axis, "blocker") == "warning"

    for name in ("correctness", "structure"):
        axis = AXES.create(name)
        assert clamp_severity(AXES.create(name), "blocker") == "blocker"
        assert "blocker" in allowed_severities(axis)


def test_reviewer_caps_an_over_severe_finding_and_records_it():
    payload = json.dumps(
        {
            "axis": "scope",
            "findings": [
                {
                    "severity": "blocker",
                    "file": "-",
                    "line": None,
                    "issue": "the PR title claims code changes but the diff is docs only",
                    "evidence": "only .gitignore and docs/ changed",
                    "reasoning": "scope mismatch",
                    "confidence": 0.99,
                }
            ],
        }
    )
    cfg = _config()
    stub = _StubDispatcher(payload)
    findings = Reviewer(cfg, ModelSelector(cfg), stub).run(
        _task("scope"), ReviewContext(task=_task("scope")), _pr()
    )

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].extra["severity_capped_from"] == "blocker"


def test_whole_pr_axis_is_told_not_to_invent_a_file():
    cfg = _config()
    stub = _StubDispatcher('{"axis": "scope", "findings": []}')
    Reviewer(cfg, ModelSelector(cfg), stub).run(
        _task("scope"), ReviewContext(task=_task("scope")), _pr()
    )
    prompt = stub.prompts[0]
    assert "pull request as a WHOLE" in prompt
    assert '"file" to "-"' in prompt
    # and the severity ceiling is stated rather than left to chance
    assert '"severity": "info|warning"' in prompt


def test_file_scoped_axis_still_asks_for_a_file():
    cfg = _config()
    stub = _StubDispatcher('{"axis": "correctness", "findings": []}')
    Reviewer(cfg, ModelSelector(cfg), stub).run(
        _task("correctness"), ReviewContext(task=_task("correctness")), _pr()
    )
    assert "changed file the problem is in" in stub.prompts[0]


# --------------------------------------------------------------------------- #
# 2. Unanchored evidence is not near-certain
# --------------------------------------------------------------------------- #


def test_unanchored_evidence_caps_confidence(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    finding = Finding(
        axis="scope", severity="warning", file=".gitignore", line=None,
        issue="scope mismatch",
        evidence="The diff shows changes only to .gitignore and the docs.",
        confidence=0.99,
    )
    det = DeterministicVerifier().verify(
        finding, repo_root=tmp_path, pr=_pr(), analysis=_analysis(), meta=None
    )
    assert EVIDENCE_NOT_ANCHORED in det.checks
    assert det.verdict == Verdict.UNCERTAIN

    VerificationRouter(_config()).adjust(finding, det)
    assert finding.confidence == UNANCHORED_CONFIDENCE_CAP
    assert finding.extra["confidence_capped_from"] == 0.99
    assert finding.extra["evidence_anchored"] is False


def test_whole_pr_finding_is_not_penalised_for_having_no_file(tmp_path: Path):
    finding = Finding(
        axis="scope", severity="warning", file="-", line=None,
        issue="scope mismatch", evidence="docs only", confidence=0.7,
    )
    det = DeterministicVerifier().verify(
        finding, repo_root=tmp_path, pr=_pr(), analysis=_analysis(), meta=None
    )
    assert det.checks == [NO_FILE_CITED]
    VerificationRouter(_config()).adjust(finding, det)
    assert finding.confidence == 0.7  # absence of a file is not evidence against


# --------------------------------------------------------------------------- #
# 3. The LLM verifier is told what the cheap pass found
# --------------------------------------------------------------------------- #


@pytest.fixture
def _verifier_probe():
    cfg = _config()
    stub = _StubDispatcher(
        '{"verdict": "invalid", "evidence": "the diff does change source files"}'
    )
    return cfg, stub, LLMVerifier(cfg, ModelSelector(cfg), stub)


def test_llm_verifier_receives_the_unanchored_warning(_verifier_probe, tmp_path: Path):
    _cfg, stub, verifier = _verifier_probe
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    finding = Finding(
        axis="scope", severity="warning", file=".gitignore", line=None,
        issue="scope mismatch", evidence="a sentence that is not in the file",
        confidence=0.6,
    )
    prior = VerdictResult(
        Verdict.UNCERTAIN, method="deterministic", checks=[EVIDENCE_NOT_ANCHORED]
    )
    result = verifier.verify(
        finding, repo_root=tmp_path, pr=_pr(), analysis=_analysis(), meta=None, prior=prior
    )
    assert "PRE-CHECKS:" in stub.prompts[0]
    assert "does NOT appear in the file it cited" in stub.prompts[0]
    assert result.verdict == Verdict.INVALID


def test_whole_pr_claim_is_verified_against_the_change_not_a_file_slice(
    _verifier_probe, tmp_path: Path
):
    _cfg, stub, verifier = _verifier_probe
    finding = Finding(
        axis="scope", severity="warning", file="-", line=None,
        issue="the PR claims code changes but only docs changed",
        evidence="only .gitignore and docs/ appear in the diff", confidence=0.6,
    )
    verifier.verify(
        finding, repo_root=tmp_path, pr=_pr(), analysis=_analysis(), meta=None, prior=None
    )
    prompt = stub.prompts[0]
    assert "COMPLETE LIST OF FILES CHANGED BY THIS PR" in prompt
    assert "docs/configuration.md" in prompt
    assert "RELEVANT SOURCE" not in prompt
    assert "misdescribes what the PR" in prompt


# --------------------------------------------------------------------------- #
# 4. The root every later stage joins paths onto
# --------------------------------------------------------------------------- #


def test_a_subdirectory_root_is_lifted_to_the_git_top_level(tiny_repo: Path):
    """`git diff` paths are top-level relative, so the root must be too.

    Pointing --repo at a subdirectory used to leave every `root / path` join
    one level too deep. Nothing raised: the review simply ran on the diff with
    no file bodies, and every finding came back unanchored.
    """
    pr, root = LocalGitSource(
        tiny_repo / "src", base_ref="main", head_ref="feature"
    ).load()

    assert root == tiny_repo.resolve()
    assert pr.extra["repo_root_adjusted"] == str(tiny_repo.resolve())
    assert "src/calc.py" in pr.changed_paths
    # the join the rest of the pipeline performs now lands on a real file
    assert (root / "src/calc.py").is_file()


def test_root_is_left_alone_when_it_already_is_the_top_level(tiny_repo: Path):
    pr, root = LocalGitSource(tiny_repo, base_ref="main", head_ref="feature").load()
    assert root == tiny_repo.resolve()
    assert pr.extra["repo_root_adjusted"] == ""


def test_findings_anchor_when_the_review_started_from_a_subdirectory(tiny_repo: Path):
    _pr, root = LocalGitSource(
        tiny_repo / "tests", base_ref="main", head_ref="feature"
    ).load()
    finding = Finding(
        axis="correctness", severity="warning", file="src/calc.py", line=None,
        issue="divide has no zero check", evidence="def divide(a, b):", confidence=0.8,
    )
    det = DeterministicVerifier().verify(
        finding, repo_root=root, pr=_pr, analysis=_analysis(), meta=None
    )
    assert det.verdict == Verdict.VALID


# --------------------------------------------------------------------------- #
# 5. What counts as quoting the file
# --------------------------------------------------------------------------- #

_SOURCE = """class PaymentService:
    def validate_request(self, req):
        if req.currency.upper() not in ("USD", "EUR", "TRY", "GBP"):
            raise ValueError("unsupported currency")

    def calculate_fee(self, amount, currency):
        if currency == "TRY":
            return amount * 0.02
        return amount * 0.01
"""


@pytest.mark.parametrize(
    "evidence, anchored, judged",
    [
        # a plain single-line quote
        ('if currency == "TRY":', True, True),
        # the shape a model actually produces: several lines, elided
        (
            'if req.currency.upper() not in ("USD", "EUR", "TRY", "GBP"): ...\n'
            'if currency == "TRY": ...',
            True,
            True,
        ),
        # re-indented to fit the reply
        ('        if currency == "TRY":', True, True),
        # fenced, as models like to do
        ("`return amount * 0.02`", True, True),
        # a narrative sentence -- the actual failure mode being guarded
        ("The service validates GBP but never prices it.", False, True),
        # too short to be distinctive: unverifiable, but NOT evidence of a lie
        ("return", False, False),
    ],
)
def test_quote_detection(tmp_path: Path, evidence: str, anchored: bool, judged: bool):
    (tmp_path / "payment_service.py").write_text(_SOURCE)
    finding = Finding(
        axis="correctness", severity="warning", file="payment_service.py", line=None,
        issue="GBP is accepted but not priced", evidence=evidence, confidence=0.97,
    )
    det = DeterministicVerifier().verify(
        finding, repo_root=tmp_path, pr=_pr(), analysis=_analysis(), meta=None
    )
    assert (det.verdict == Verdict.VALID) is anchored
    assert (EVIDENCE_NOT_ANCHORED in det.checks) is (judged and not anchored)

    VerificationRouter(_config()).adjust(finding, det)
    if judged and not anchored:
        assert finding.confidence == UNANCHORED_CONFIDENCE_CAP
    else:
        assert finding.confidence == 0.97  # untouched
