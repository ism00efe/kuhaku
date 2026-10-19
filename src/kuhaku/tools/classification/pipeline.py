"""Composable classification pipeline -- :mod:`.steps_base` with
classification's policy.

Every stage around the model's decision -- deterministic rules in front of it, caller
validators after it, and later sanitization or the input guard -- is a :class:`Step`. A
pipeline is an ordered list of named steps that can be added, removed, replaced,
enabled or disabled. The one step that cannot be removed or disabled is ``classify``:
it is what guarantees every text ends decided or undecided. It can still be *replaced*
by another step of the same name.

A run has two phases. ``prepare`` runs for every enabled step first, so any step can
contribute a prompt rule before the model is called. ``run`` then executes in order and
stops early once a step calls :meth:`ClassificationContext.stop`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from .context import ClassificationContext
from .steps_base import Pipeline as CorePipeline
from .steps_base import Step as CoreStep

CLASSIFY_STEP = "classify"


logger = logging.getLogger(__name__)


class Step(CoreStep[ClassificationContext]):
    """Base class for a classification pipeline stage. Override either hook, or both."""


class Pipeline(CorePipeline[ClassificationContext]):
    def __init__(self, steps: Iterable[CoreStep[ClassificationContext]]) -> None:
        super().__init__(
            steps,
            required={CLASSIFY_STEP: "it guarantees every text ends decided or undecided"},
            metric_prefix="classification",
            label="classification",
            log=logger,
        )

    def step_fields(self, ctx: ClassificationContext) -> dict[str, Any]:
        return {"outcome": ctx.status.value if ctx.status else None}
