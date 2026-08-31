"""The adaptive review pipeline.

    understand the change
      -> decide relevant review axes
      -> decide depth per axis
      -> gather only the necessary context
      -> choose an appropriate model
      -> generate findings
      -> verify important / uncertain findings
      -> produce the final review

Every stage is resolved from a registry keyed by a name in :class:`Config`.
This module contains no axis, depth, provider or verifier name -- adding one is
purely additive.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Importing these packages populates every plugin registry as a side effect.
from pr_review import analyzer as _analyzer_pkg  # noqa: F401
from pr_review import axes as _axes_pkg  # noqa: F401
from pr_review import context as _context_pkg  # noqa: F401
from pr_review import depths as _depths_pkg  # noqa: F401
from pr_review import discovery as _discovery_pkg  # noqa: F401
from pr_review import providers as _providers_pkg  # noqa: F401
from pr_review import report as _report_pkg  # noqa: F401
from pr_review import verification as _verification_pkg  # noqa: F401
from pr_review.analyzer.classifier import Classifier
from pr_review.analyzer.deterministic import ANALYZERS
from pr_review.config import Config
from pr_review.context.base import CONTEXT_STRATEGIES
from pr_review.discovery.base import DISCOVERERS, merge_metadata
from pr_review.errors import PRReviewError
from pr_review.models import (
    Finding,
    PRContext,
    RepoMetadata,
    ReviewResult,
    ReviewTask,
    StageError,
    Verdict,
    VerdictResult,
    VerifiedFinding,
)
from pr_review.plan.planner import Planner
from pr_review.providers.dispatch import Dispatcher
from pr_review.providers.selector import ModelSelector
from pr_review.review.reviewer import Reviewer
from pr_review.source.base import PRSource
from pr_review.verification.base import VERIFIERS
from pr_review.verification.router import VerificationRouter


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.selector = ModelSelector(config)
        self.dispatcher = Dispatcher(config, self.selector)

    # ------------------------------------------------------------------ #

    def run(self, source: PRSource) -> ReviewResult:
        t0 = time.time()
        errors: list[StageError] = []

        pr, repo_root = source.load()

        meta = self._discover(repo_root, errors)
        meta.root = str(repo_root)

        analysis = self._analyze(pr, meta, errors)

        # With no reachable model the run is still useful -- discovery, the
        # deterministic analysis and the heuristic plan all stand on their own.
        # What it must not do is manufacture findings, so the LLM stages are
        # skipped outright and the report says so.
        llm_available = self.config.has_llm()

        classifier = Classifier(self.config, self.selector, self.dispatcher)
        plan = classifier.plan(analysis, meta, pr)

        tasks = Planner(self.config).build(plan)

        if llm_available:
            raw_findings = self._review(tasks, repo_root, pr, analysis, meta, errors)
            verified = self._verify(raw_findings, repo_root, pr, analysis, meta, errors)
        else:
            raw_findings = []
            verified = []
        verified = self._dedupe(verified)[: self.config.max_total_findings]

        stats = {
            "mode": "llm" if llm_available else "structural-only",
            "duration_seconds": round(time.time() - t0, 2),
            "tasks_run": len(tasks),
            "raw_findings": len(raw_findings),
            "surfaced_findings": sum(
                1 for vf in verified if vf.result.verdict != Verdict.INVALID
            ),
            "plan_source": plan.source,
            "llm_verifications": sum(
                1 for vf in verified if vf.result.method == "llm"
            ),
        }
        return ReviewResult(
            pr=pr, repo=meta, analysis=analysis, plan=plan, tasks=tasks,
            findings=verified, errors=errors, notes=list(self.dispatcher.notes),
            stats=stats,
        )

    # ------------------------------------------------------------------ #

    def _discover(self, root: Path, errors: list[StageError]) -> RepoMetadata:
        parts: list[RepoMetadata] = []
        for name in self.config.discoverers:
            if not DISCOVERERS.has(name):
                errors.append(StageError("discovery", f"unknown discoverer {name!r}"))
                continue
            try:
                parts.append(DISCOVERERS.create(name).discover(root))
            except Exception as exc:  # noqa: BLE001
                errors.append(StageError("discovery", f"{name}: {exc}"))
        return merge_metadata(parts, root)

    def _analyze(self, pr: PRContext, meta: RepoMetadata, errors: list[StageError]):
        analyzers = self.config.deterministic_analyzers or ("core",)
        result = None
        for name in analyzers:
            try:
                a = ANALYZERS.create(name).analyze(pr, meta)
            except Exception as exc:  # noqa: BLE001
                errors.append(StageError("analysis", f"{name}: {exc}"))
                continue
            result = a  # last analyzer wins; typically just "core"
        if result is None:
            from pr_review.models import DeterministicAnalysis

            errors.append(StageError("analysis", "no analyzer produced output"))
            return DeterministicAnalysis()
        return result

    def _review(
        self,
        tasks: list[ReviewTask],
        root: Path,
        pr: PRContext,
        analysis,
        meta: RepoMetadata,
        errors: list[StageError],
    ) -> list[Finding]:
        gatherer = CONTEXT_STRATEGIES.create(self.config.context_strategy)
        reviewer = Reviewer(self.config, self.selector, self.dispatcher)
        delay = self.config.concurrency.axis_delay_seconds

        def one(task: ReviewTask) -> list[Finding]:
            try:
                ctx = gatherer.gather(
                    task, repo_root=root, pr=pr, analysis=analysis, meta=meta
                )
                return reviewer.run(task, ctx, pr)
            except PRReviewError as exc:
                errors.append(
                    StageError("review", f"{task.axis}/{task.depth}: {exc}")
                )
                return []
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    StageError("review", f"{task.axis}/{task.depth}: unexpected {exc}")
                )
                return []

        findings: list[Finding] = []
        workers = max(1, self.config.concurrency.workers)
        if workers == 1:
            for i, task in enumerate(tasks):
                if i and delay:
                    time.sleep(delay)
                findings.extend(one(task))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for res in ex.map(one, tasks):
                    findings.extend(res)
        return findings

    def _verify(
        self,
        findings: list[Finding],
        root: Path,
        pr: PRContext,
        analysis,
        meta: RepoMetadata,
        errors: list[StageError],
    ) -> list[VerifiedFinding]:
        if not self.config.verification.enabled or not findings:
            return [
                VerifiedFinding(f, VerdictResult(Verdict.UNCERTAIN, method="skipped"))
                for f in findings
            ]

        det = None
        if "deterministic" in self.config.verifiers and VERIFIERS.has("deterministic"):
            det = VERIFIERS.create("deterministic")
        llm = None
        if (
            self.config.verification.llm_enabled
            and "llm" in self.config.verifiers
            and VERIFIERS.has("llm")
        ):
            try:
                llm = VERIFIERS.create(
                    "llm", self.config, self.selector, dispatcher=self.dispatcher
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(StageError("verification", f"llm init: {exc}"))

        router = VerificationRouter(self.config)
        ordered = router.order(findings)
        out: list[VerifiedFinding] = []
        llm_budget = self.config.verification.max_llm_verifications

        for f in ordered:
            result = VerdictResult(Verdict.UNCERTAIN, method="skipped")
            if det is not None:
                try:
                    result = det.verify(
                        f, repo_root=root, pr=pr, analysis=analysis, meta=meta
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(StageError("verification", f"deterministic: {exc}"))
            # What the cheap pass learned shapes both the finding's confidence
            # and what the expensive pass is told.
            router.adjust(f, result)
            if llm is not None and llm_budget > 0 and router.needs_llm(f, result):
                llm_budget -= 1
                try:
                    llm_res = llm.verify(
                        f, repo_root=root, pr=pr, analysis=analysis, meta=meta,
                        prior=result,
                    )
                    if llm_res.verdict != Verdict.UNCERTAIN or result.verdict == Verdict.UNCERTAIN:
                        llm_res.checks = result.checks + llm_res.checks
                        result = llm_res
                except Exception as exc:  # noqa: BLE001
                    errors.append(StageError("verification", f"llm: {exc}"))
            out.append(VerifiedFinding(f, result))
        return out

    @staticmethod
    def _dedupe(verified: list[VerifiedFinding]) -> list[VerifiedFinding]:
        seen: dict[tuple, VerifiedFinding] = {}
        for vf in verified:
            key = vf.finding.key()
            cur = seen.get(key)
            if cur is None or vf.finding.confidence > cur.finding.confidence:
                seen[key] = vf
        rank = {"blocker": 0, "warning": 1, "info": 2}
        return sorted(
            seen.values(),
            key=lambda vf: (
                rank.get(vf.finding.severity, 3),
                0 if vf.result.verdict == Verdict.VALID else 1,
                -vf.finding.confidence,
            ),
        )
