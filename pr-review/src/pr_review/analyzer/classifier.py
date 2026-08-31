"""Small-LLM change classifier + router.

Turns deterministic analysis + repo metadata + PR text into a
:class:`ReviewPlan` (which axes, at which depth). This is planning only -- it
does not review code. When no LLM is available, or its output cannot be parsed,
a transparent heuristic produces the plan instead, so the pipeline always yields
a usable plan.
"""

from __future__ import annotations

import json

from pr_review.config import Config
from pr_review.models import (
    Depth,
    DeterministicAnalysis,
    PlanEntry,
    PRContext,
    RepoMetadata,
    ReviewPlan,
)
from pr_review.providers.base import extract_json
from pr_review.providers.dispatch import Dispatcher
from pr_review.providers.selector import ModelSelector

_VALID_DEPTHS = {d.value for d in Depth}


class Classifier:
    def __init__(
        self,
        config: Config,
        selector: ModelSelector,
        dispatcher: Dispatcher | None = None,
    ) -> None:
        self.config = config
        self.selector = selector
        self.dispatcher = dispatcher or Dispatcher(config, selector)

    def plan(
        self, analysis: DeterministicAnalysis, meta: RepoMetadata, pr: PRContext
    ) -> ReviewPlan:
        heuristic = self._heuristic(analysis, meta, pr)
        if not self.config.classifier_enabled or not self.config.has_llm():
            return heuristic
        try:
            completion = self.dispatcher.complete(
                "planner",
                self._prompt(analysis, meta, pr),
                system="You are a precise triage assistant. Respond with JSON only.",
            )
            plan = self._parse(completion.text, allowed_axes=set(self.config.enabled_axes))
            if plan and plan.entries:
                # Union with heuristic-required axes so the model cannot silently
                # drop a structural risk the deterministic layer already proved.
                plan = self._reconcile(plan, heuristic)
                plan.source = "llm"
                return plan
        except Exception as exc:  # noqa: BLE001 - never let planning abort the run
            heuristic.raw = f"planner error: {exc}"
        return heuristic

    # ------------------------------------------------------------------ #

    def _prompt(
        self, a: DeterministicAnalysis, meta: RepoMetadata, pr: PRContext
    ) -> str:
        axes = ", ".join(self.config.enabled_axes)
        depths = ", ".join([Depth.OFF.value, *self.config.enabled_depths])
        files = "\n".join(
            f"  {f.status:8} +{f.additions} -{f.deletions} {f.path}"
            for f in a.files[: self.config.limits.max_changed_files_listed]
        )
        syms = ", ".join(
            f"{s.change} {s.kind} {s.name}" for s in a.changed_symbols[:25]
        )
        return f"""TASK: plan
Decide which review axes to run and how deep, for this pull request.

AVAILABLE AXES: {axes}
AVAILABLE DEPTHS: {depths}  (off = skip the axis)

REPOSITORY
  languages: {dict(sorted(meta.languages.items(), key=lambda kv: -kv[1]))}
  package_managers: {meta.package_managers}
  frameworks: {meta.frameworks}
  architecture_docs: {meta.architecture_docs}

PR TITLE: {pr.title}
PR BODY: {pr.body[:1200]}

DETERMINISTIC ANALYSIS
  changed_files: {a.changed_file_count}  (+{a.total_additions} / -{a.total_deletions})
{files}
  changed_symbols: {syms or "(none detected)"}
  added_imports: {a.added_imports[:20]}
  dependency_changes: {a.dependency_changes}
  interface_changes: {a.interface_changes}
  touched_areas: {a.touched_areas}
  signals: {json.dumps(a.signals, default=str)}

Respond with JSON only, no prose:
{{"change_type": "<short label>",
  "risk_areas": ["..."],
  "reviews": [{{"axis": "<axis>", "depth": "<depth>", "reason": "<short>"}}]}}
Include an entry only for axes worth running. Prefer the shallowest depth that
is sufficient. Escalate depth only when the change is broad, risky, or touches
interfaces/structure."""

    def _parse(self, raw: str, *, allowed_axes: set[str]) -> ReviewPlan | None:
        try:
            data = extract_json(raw)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        entries: list[PlanEntry] = []
        for item in data.get("reviews", []):
            if not isinstance(item, dict):
                continue
            axis = str(item.get("axis", "")).strip().lower()
            depth = str(item.get("depth", "")).strip().lower()
            if axis not in allowed_axes or depth not in _VALID_DEPTHS:
                continue
            entries.append(PlanEntry(axis=axis, depth=depth, reason=str(item.get("reason", ""))))
        return ReviewPlan(
            entries=entries,
            change_type=str(data.get("change_type", "")),
            risk_areas=[str(x) for x in data.get("risk_areas", []) if isinstance(x, str)],
            raw=raw[:4000],
        )

    def _reconcile(self, plan: ReviewPlan, heuristic: ReviewPlan) -> ReviewPlan:
        by_axis = {e.axis: e for e in plan.entries}
        order = {Depth.OFF.value: 0, "basic": 1, "normal": 2, "deep": 3}
        for h in heuristic.entries:
            if h.depth == Depth.OFF.value:
                continue
            cur = by_axis.get(h.axis)
            if cur is None:
                by_axis[h.axis] = h
            elif order.get(h.depth, 0) > order.get(cur.depth, 0):
                by_axis[h.axis] = PlanEntry(h.axis, h.depth, f"raised: {h.reason}")
        plan.entries = list(by_axis.values())
        if not plan.risk_areas:
            plan.risk_areas = heuristic.risk_areas
        return plan

    # ------------------------------------------------------------------ #

    def _heuristic(
        self, a: DeterministicAnalysis, meta: RepoMetadata, pr: PRContext
    ) -> ReviewPlan:
        s = a.signals
        enabled = set(self.config.enabled_axes)
        entries: dict[str, PlanEntry] = {}

        def want(axis: str, depth: str, reason: str) -> None:
            if axis not in enabled:
                return
            order = {"off": 0, "basic": 1, "normal": 2, "deep": 3}
            cur = entries.get(axis)
            if cur is None or order[depth] > order[cur.depth]:
                entries[axis] = PlanEntry(axis, depth, reason)

        if s.get("only_docs"):
            want("scope", "basic", "documentation-only change")
        else:
            want("correctness", "basic", "always inspect changed logic")
            want("scope", "basic", "check implementation matches PR intent")

        if s.get("non_trivial"):
            want("correctness", "normal", "non-trivial change; consider callers/callees")
            want("method", "basic", "assess whether the approach fits")
        if s.get("dependency_files_changed") or s.get("new_top_level_dir"):
            want("structure", "normal", "dependency or top-level structure changed")
        if s.get("interface_changed"):
            want("structure", "deep", "public interface / schema changed")
            want("correctness", "normal", "interface change ripples to callers")
        if s.get("large_change"):
            for ax in ("correctness", "method", "structure"):
                want(ax, "deep", "large change surface")
            want("scope", "normal", "large change; verify nothing unrelated slipped in")
        if s.get("symbols_removed"):
            want("structure", "normal", "symbols removed; check remaining references")

        risk = []
        if s.get("interface_changed"):
            risk.append("public interface")
        if s.get("dependency_files_changed"):
            risk.append("dependencies")
        if s.get("large_change"):
            risk.append("large surface")

        return ReviewPlan(
            entries=list(entries.values()),
            change_type=self._change_type(a, pr),
            risk_areas=risk,
            source="heuristic",
        )

    @staticmethod
    def _change_type(a: DeterministicAnalysis, pr: PRContext) -> str:
        title = pr.title.lower()
        for kw, label in (
            ("fix", "bugfix"), ("bug", "bugfix"), ("feat", "feature"),
            ("refactor", "refactor"), ("docs", "docs"), ("test", "tests"),
            ("chore", "chore"), ("perf", "performance"),
        ):
            if title.startswith(kw) or f"{kw}:" in title or f"{kw}(" in title:
                return label
        if a.signals.get("only_docs"):
            return "docs"
        if a.added_paths and not a.deleted_paths:
            return "feature"
        return "change"
