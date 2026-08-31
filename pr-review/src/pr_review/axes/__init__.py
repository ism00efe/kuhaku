"""Review axes.

An axis defines *what to look for*. It contributes prompt text and hints about
what context is valuable; it does not know about models or depths. Add an axis
by creating a module here and registering it -- nothing else changes.
"""

from pr_review.axes import correctness, method, scope, structure  # noqa: F401
from pr_review.axes.base import AXES, ReviewAxis

__all__ = ["AXES", "ReviewAxis"]
