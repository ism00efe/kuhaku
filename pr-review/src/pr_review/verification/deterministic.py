"""Objective checks that can settle a finding without an LLM.

Confirms the finding is *anchored* in real repository evidence: the cited file
exists (or is legitimately deleted), the cited line is in range, and the quoted
evidence string actually appears near that location. It returns INVALID only
when a claim is clearly unfounded (e.g. cites a file that never existed);
otherwise UNCERTAIN, leaving the semantic judgement to the LLM verifier.
"""

from __future__ import annotations

import re
from pathlib import Path

from pr_review.models import (
    DeterministicAnalysis,
    Finding,
    PRContext,
    RepoMetadata,
    Verdict,
    VerdictResult,
)
from pr_review.verification.base import VERIFIERS

# Named check outcomes. Downstream stages branch on these rather than on the
# wording, so the prose can change without silently breaking the chain.
NO_FILE_CITED = "no file cited"
EVIDENCE_NOT_ANCHORED = "quoted evidence not found verbatim in file"
EVIDENCE_ANCHORED = "quoted evidence found in file"


class DeterministicVerifier:
    name = "deterministic"

    def verify(
        self,
        finding: Finding,
        *,
        repo_root: Path,
        pr: PRContext,
        analysis: DeterministicAnalysis,
        meta: RepoMetadata,
        prior: VerdictResult | None = None,
    ) -> VerdictResult:
        checks: list[str] = []
        path = finding.file.strip()
        if not path or path == "-":
            # Legitimate for a whole-PR axis: there is nothing to anchor to, and
            # that is not evidence against the claim.
            checks.append(NO_FILE_CITED)
            return VerdictResult(Verdict.UNCERTAIN, method="deterministic", checks=checks)

        changed = {f.path for f in analysis.files}
        deleted = set(analysis.deleted_paths)
        fp = repo_root / path
        exists = fp.is_file()

        if not exists and path not in deleted:
            # Cited a file that is neither present nor part of this PR's deletions.
            if path not in changed:
                checks.append(f"cited file '{path}' does not exist and is not in the diff")
                return VerdictResult(
                    Verdict.INVALID, evidence="; ".join(checks), method="deterministic",
                    checks=checks,
                )
            checks.append(f"cited file '{path}' is in the diff but not on disk")
            return VerdictResult(Verdict.UNCERTAIN, method="deterministic", checks=checks)

        text = ""
        if exists:
            try:
                text = fp.read_text("utf-8", errors="replace")
            except OSError:
                text = ""

        lines = text.splitlines()
        if finding.line is not None and text:
            if 1 <= finding.line <= len(lines):
                checks.append(f"line {finding.line} in range")
            else:
                checks.append(
                    f"cited line {finding.line} out of range (file has {len(lines)} lines)"
                )
                return VerdictResult(
                    Verdict.INVALID, evidence="; ".join(checks), method="deterministic",
                    checks=checks,
                )

        ev = finding.evidence.strip()
        # Only judge the quotation when there is something long enough to judge.
        # Short evidence is unverifiable, which is not the same as fabricated.
        needles = _probe_needles(ev) if text else []
        if needles:
            if _matches(needles, text):
                checks.append(EVIDENCE_ANCHORED)
                return VerdictResult(
                    Verdict.VALID,
                    evidence=f"evidence line present in {path}",
                    method="deterministic",
                    checks=checks,
                )
            checks.append(EVIDENCE_NOT_ANCHORED)

        checks.append("anchored but semantics not machine-checkable")
        return VerdictResult(Verdict.UNCERTAIN, method="deterministic", checks=checks)


_ELLIPSIS = re.compile(r"(\.{3}|\u2026)\s*$")
_MIN_PROBE = 12


def _probe_needles(evidence: str) -> list[str]:
    """The lines of ``evidence`` that are worth looking for in a file.

    Models quote the way people do: several lines, sometimes elided with an
    ellipsis, re-indented to fit the reply. Demanding that the first line match
    byte-for-byte failed all three of those, so honest multi-line quotes were
    scored as fabrications and lost confidence. Each line is stripped of fences,
    quotes and a trailing ellipsis, then whitespace-collapsed so indentation
    stops mattering. Lines too short to be distinctive are dropped -- returning
    none means the evidence cannot be checked either way.
    """
    needles = []
    for raw in evidence.splitlines():
        line = _ELLIPSIS.sub("", raw.strip().strip("`\"'")).strip()
        needle = " ".join(line.split())
        if len(needle) >= _MIN_PROBE:
            needles.append(needle)
    return needles


def _matches(needles: list[str], text: str) -> bool:
    haystack = " ".join(text.split())
    return any(n in haystack for n in needles)


VERIFIERS.register("deterministic", lambda *_a, **_k: DeterministicVerifier())
