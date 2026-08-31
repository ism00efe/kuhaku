"""Review depths.

A depth defines *how far to look* and *how strong a model to use*. It is
independent of the axis. Add a depth by registering it and listing it in
``enabled_depths``.
"""

from pr_review.depths import basic, deep, normal  # noqa: F401
from pr_review.depths.base import DEPTHS, ReviewDepth

__all__ = ["DEPTHS", "ReviewDepth"]
