"""Convert a :class:`ReviewPlan` into concrete :class:`ReviewTask` objects.

This is the one place axis + depth + model + context-strategy are bound
together. The final ``ContextSpec`` for a task is the depth's spec narrowed by
the axis's hints, so e.g. ``scope`` never pulls callers even at ``deep`` while
``structure`` does.

It is also where a change too large for one request becomes several passes.
The alternative -- one request carrying as much as fits -- silently drops the
remainder, which is worse than useless: the review reads as complete while
most of the change was never seen. Splitting keeps every file reviewed and
makes the cost explicit, since a pass is a request and requests are what the
free tiers actually meter.
"""

from __future__ import annotations

from dataclasses import replace

from pr_review.axes.base import AXES
from pr_review.config import Config
from pr_review.depths.base import DEPTHS
from pr_review.models import (
    ContextSpec,
    DeterministicAnalysis,
    PlanEntry,
    ReviewPlan,
    ReviewTask,
)

_FALLBACK_BUDGET = 24_000


class Planner:
    def __init__(self, config: Config) -> None:
        self.config = config

    def build(
        self, plan: ReviewPlan, analysis: DeterministicAnalysis | None = None
    ) -> list[ReviewTask]:
        tasks: list[ReviewTask] = []
        for entry in plan.active():
            if not AXES.has(entry.axis) or not DEPTHS.has(entry.depth):
                continue
            if entry.axis not in self.config.enabled_axes:
                continue
            if entry.depth not in self.config.enabled_depths:
                continue
            tasks.extend(self._tasks(entry, analysis))
        # Deterministic ordering: by depth strength then axis name, so free-tier
        # sequential runs spend the early budget on the cheap axes.
        weight = {"basic": 0, "normal": 1, "deep": 2}
        tasks.sort(key=lambda t: (weight.get(t.depth, 3), t.axis, t.pass_index))
        return tasks

    # ------------------------------------------------------------------ #

    def _tasks(
        self, entry: PlanEntry, analysis: DeterministicAnalysis | None
    ) -> list[ReviewTask]:
        axis = AXES.create(entry.axis)
        depth = DEPTHS.create(entry.depth)
        tier = self.config.depth_tier.get(entry.depth, depth.default_model_tier)
        spec = self._narrow(depth.context_spec(), axis.hints())
        budget = self._budget(tier)

        base = ReviewTask(
            axis=entry.axis,
            depth=entry.depth,
            model_tier=tier,
            context_strategy=self.config.context_strategy,
            context_spec=spec,
            input_budget_bytes=budget,
        )
        groups = self._split(analysis, budget)
        if len(groups) <= 1:
            return [base]
        return [
            replace(base, files=tuple(g), pass_index=i + 1, pass_count=len(groups))
            for i, g in enumerate(groups)
        ]

    def _budget(self, tier: str) -> int:
        chain = self.config.resolve_chain(tier)
        # Built for the candidate we expect to use. A smaller fallback that
        # cannot take it answers 413, which the dispatcher treats as permanent
        # and fails past -- wasteful, but never silently wrong.
        return chain[0].input_budget_bytes if chain else _FALLBACK_BUDGET

    def _split(
        self, analysis: DeterministicAnalysis | None, budget: int
    ) -> list[list[str]]:
        """Pack the changed files into passes that each fit one request."""
        if analysis is None or not analysis.files:
            return []
        total = sum(f.diff_bytes for f in analysis.files)
        if total <= budget:
            return []  # one pass over the whole change

        groups: list[list[str]] = []
        current: list[str] = []
        used = 0
        # Path order keeps a pass coherent: files that live together are read
        # together, and the split is reproducible across runs.
        for f in sorted(analysis.files, key=lambda f: f.path):
            if current and used + f.diff_bytes > budget:
                groups.append(current)
                current, used = [], 0
            current.append(f.path)
            used += f.diff_bytes
        if current:
            groups.append(current)

        cap = self.config.limits.max_passes_per_axis
        if cap and len(groups) > cap:
            groups = groups[:cap]
        return groups

    @staticmethod
    def _narrow(spec: ContextSpec, hints) -> ContextSpec:
        return replace(
            spec,
            sibling_files=spec.sibling_files and hints.wants_sibling_files,
            callers_usages=spec.callers_usages and hints.wants_callers_usages,
            dependency_files=spec.dependency_files and hints.wants_dependency_files,
            architecture_docs=spec.architecture_docs and hints.wants_architecture_docs,
            repo_tree=spec.repo_tree and hints.wants_repo_tree,
        )
