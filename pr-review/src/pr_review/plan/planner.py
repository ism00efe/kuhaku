"""Convert a :class:`ReviewPlan` into concrete :class:`ReviewTask` objects.

This is the one place axis + depth + model + context-strategy are bound
together. The final ``ContextSpec`` for a task is the depth's spec narrowed by
the axis's hints, so e.g. ``scope`` never pulls callers even at ``deep`` while
``structure`` does.
"""

from __future__ import annotations

from dataclasses import replace

from pr_review.axes.base import AXES
from pr_review.config import Config
from pr_review.depths.base import DEPTHS
from pr_review.models import ContextSpec, PlanEntry, ReviewPlan, ReviewTask


class Planner:
    def __init__(self, config: Config) -> None:
        self.config = config

    def build(self, plan: ReviewPlan) -> list[ReviewTask]:
        tasks: list[ReviewTask] = []
        for entry in plan.active():
            if not AXES.has(entry.axis) or not DEPTHS.has(entry.depth):
                continue
            if entry.axis not in self.config.enabled_axes:
                continue
            if entry.depth not in self.config.enabled_depths:
                continue
            tasks.append(self._task(entry))
        # Deterministic ordering: by depth strength then axis name, so free-tier
        # sequential runs spend the early budget on the cheap axes.
        weight = {"basic": 0, "normal": 1, "deep": 2}
        tasks.sort(key=lambda t: (weight.get(t.depth, 3), t.axis))
        return tasks

    def _task(self, entry: PlanEntry) -> ReviewTask:
        axis = AXES.create(entry.axis)
        depth = DEPTHS.create(entry.depth)
        tier = self.config.depth_tier.get(entry.depth, depth.default_model_tier)
        spec = self._narrow(depth.context_spec(), axis.hints())
        return ReviewTask(
            axis=entry.axis,
            depth=entry.depth,
            model_tier=tier,
            context_strategy=self.config.context_strategy,
            context_spec=spec,
        )

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
