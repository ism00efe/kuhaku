"""Finding verification.

Detection produces *claims*; verification decides whether the repository
actually supports each claim. Deterministic checks run first (free); the LLM
verifier runs only on the findings the router still considers important or
unresolved.
"""

from pr_review.verification import deterministic, llm  # noqa: F401
from pr_review.verification.base import VERIFIERS, Verifier
from pr_review.verification.router import VerificationRouter

__all__ = ["VERIFIERS", "Verifier", "VerificationRouter"]
