"""Repository-agnostic, modular AI pull-request review engine.

Public surface is intentionally small: build a :class:`~pr_review.config.Config`,
pick a :class:`~pr_review.source.base.PRSource`, and call
:meth:`~pr_review.pipeline.Pipeline.run`.
"""

from pr_review.config import Config
from pr_review.pipeline import Pipeline

__version__ = "0.1.0"

__all__ = ["Config", "Pipeline", "__version__"]
