"""Repository discovery.

Independent discoverers each contribute part of a :class:`RepoMetadata`; the
pipeline merges their output. None of them assume a language or layout -- they
detect. Add one by registering it and adding its name to ``discoverers`` in
config.
"""

from pr_review.discovery import languages, layout, manifests  # noqa: F401
from pr_review.discovery.base import DISCOVERERS, RepoDiscoverer, merge_metadata

__all__ = ["DISCOVERERS", "RepoDiscoverer", "merge_metadata"]
