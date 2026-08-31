"""Depth contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pr_review.models import ContextSpec
from pr_review.registry import Registry


@runtime_checkable
class ReviewDepth(Protocol):
    name: str
    default_model_tier: str

    def context_spec(self) -> ContextSpec:
        """The upper bound of context this depth is willing to gather.

        The context gatherer intersects this with the axis's hints.
        """
        ...

    def instruction(self) -> str:
        """Extra guidance appended to the reviewer prompt for this depth."""
        ...


DEPTHS: Registry[ReviewDepth] = Registry("depth")
