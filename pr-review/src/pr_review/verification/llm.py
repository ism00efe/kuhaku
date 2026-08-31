"""Strong-model verifier for important or ambiguous findings.

Asks a strong model whether the repository evidence actually supports a claim.
Two things keep this from becoming a rubber stamp:

* it is told what the deterministic pass already established -- in particular
  when the "quoted" evidence does not appear in the cited file at all;
* a claim about the pull request as a whole (the ``scope`` and ``method`` axes)
  is judged against the complete changed-file list and the PR text, not against
  a slice of some file the model happened to name. Slicing an unrelated file
  and asking "is this true?" invites a yes.

Used sparingly -- the router decides which findings reach here and caps the
count.
"""

from __future__ import annotations

from pathlib import Path

from pr_review.config import Config
from pr_review.errors import ProviderError
from pr_review.models import (
    DeterministicAnalysis,
    Finding,
    PRContext,
    RepoMetadata,
    Verdict,
    VerdictResult,
)
from pr_review.providers.base import extract_json
from pr_review.providers.dispatch import Dispatcher
from pr_review.providers.selector import ModelSelector
from pr_review.verification.base import VERIFIERS
from pr_review.verification.deterministic import (
    EVIDENCE_ANCHORED,
    EVIDENCE_NOT_ANCHORED,
)

_SYSTEM = (
    "You verify code-review findings against evidence. You are skeptical: a "
    "finding is VALID only if the provided evidence clearly supports it. A "
    "restatement of the claim is not evidence for it. Reply with a single JSON "
    "object."
)

_PRECHECK_NOTES = {
    EVIDENCE_NOT_ANCHORED: (
        "A deterministic pre-check found that the text the reviewer quoted as "
        "evidence does NOT appear in the file it cited. Treat the claim as "
        "unsupported unless the material below independently establishes it."
    ),
    EVIDENCE_ANCHORED: (
        "A deterministic pre-check confirmed the quoted line does appear in the "
        "cited file. That settles the quotation only, not the conclusion."
    ),
}


class LLMVerifier:
    name = "llm"

    def __init__(
        self,
        config: Config,
        selector: ModelSelector,
        dispatcher: Dispatcher | None = None,
    ) -> None:
        self.config = config
        self.selector = selector
        self.dispatcher = dispatcher or Dispatcher(config, selector)

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
        prompt = self._prompt(finding, repo_root, pr, analysis, prior)

        try:
            completion = self.dispatcher.complete(
                "verify", prompt, system=_SYSTEM, temperature=0.0
            )
            data = extract_json(completion.text)
        except (ProviderError, ValueError) as exc:
            return VerdictResult(
                Verdict.UNCERTAIN, evidence=f"verifier unavailable: {exc}", method="llm"
            )

        if not isinstance(data, dict):
            return VerdictResult(Verdict.UNCERTAIN, method="llm")
        verdict = str(data.get("verdict", "uncertain")).lower()
        if verdict not in {v.value for v in Verdict}:
            verdict = "uncertain"
        return VerdictResult(
            Verdict(verdict),
            evidence=str(data.get("evidence", ""))[:600],
            method="llm",
        )

    # ------------------------------------------------------------------ #

    def _prompt(
        self,
        finding: Finding,
        repo_root: Path,
        pr: PRContext,
        analysis: DeterministicAnalysis,
        prior: VerdictResult | None,
    ) -> str:
        claim = f"""TASK: verify
A reviewer made this claim about a pull request:

  axis: {finding.axis}
  severity: {finding.severity}
  file: {finding.file}
  line: {finding.line}
  issue: {finding.issue}
  evidence cited: {finding.evidence}
  reasoning: {finding.reasoning}
"""
        notes = [
            _PRECHECK_NOTES[c]
            for c in (prior.checks if prior else [])
            if c in _PRECHECK_NOTES
        ]
        precheck = ("\nPRE-CHECKS:\n" + "\n".join(f"- {n}" for n in notes)) if notes else ""

        if finding.file.strip() in ("", "-"):
            material = f"""COMPLETE LIST OF FILES CHANGED BY THIS PR:
{self._changed_files(analysis)}

PR TITLE: {pr.title}
PR BODY: {pr.body[:1200]}

This claim is about the pull request as a whole. Judge it ONLY against the
changed-file list and the PR text above. If the claim misdescribes what the PR
actually changes, answer invalid."""
        else:
            material = f"""RELEVANT SOURCE (from {finding.file}):
```
{self._slice(repo_root / finding.file, finding.line)}
```

PR TITLE: {pr.title}"""

        return f"""{claim}{precheck}

{material}

Decide whether the evidence supports the claim. Respond with ONLY:
{{"verdict": "valid|invalid|uncertain",
  "evidence": "<one or two sentences citing what you saw>"}}"""

    @staticmethod
    def _changed_files(analysis: DeterministicAnalysis, limit: int = 60) -> str:
        rows = [
            f"  {f.status:8} {f.path} (+{f.additions}/-{f.deletions})"
            for f in analysis.files[:limit]
        ]
        if len(analysis.files) > limit:
            rows.append(f"  ... and {len(analysis.files) - limit} more")
        return "\n".join(rows) or "  (none)"

    @staticmethod
    def _slice(path: Path, line: int | None, radius: int = 40) -> str:
        try:
            lines = path.read_text("utf-8", errors="replace").splitlines()
        except OSError:
            return "(file not readable)"
        if not lines:
            return "(empty file)"
        if line is None:
            return "\n".join(lines[:120])
        lo = max(0, line - radius)
        hi = min(len(lines), line + radius)
        return "\n".join(f"{i + 1:5} {lines[i]}" for i in range(lo, hi))


VERIFIERS.register(
    "llm",
    lambda config, selector, dispatcher=None, **_k: LLMVerifier(
        config, selector, dispatcher
    ),
)
