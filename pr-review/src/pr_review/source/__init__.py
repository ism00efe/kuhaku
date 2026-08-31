"""PR sources: adapters that turn some external world into a :class:`PRContext`.

The review engine never imports anything from this package; the CLI wires a
source in. This is the seam that keeps GitHub (or any future host) swappable.
"""

from pr_review.source.base import PRSource
from pr_review.source.github import GitHubActionsSource
from pr_review.source.local_git import LocalGitSource

__all__ = ["PRSource", "GitHubActionsSource", "LocalGitSource"]
