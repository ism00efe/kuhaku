"""Context gathering strategies.

Given a task (axis + depth + spec), produce the smallest context sufficient for
that task. Registered by name; select via ``context_strategy`` in config.
"""

from pr_review.context import default  # noqa: F401
from pr_review.context.base import CONTEXT_STRATEGIES, ContextGatherer

__all__ = ["CONTEXT_STRATEGIES", "ContextGatherer"]
