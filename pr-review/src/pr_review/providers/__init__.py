"""LLM providers.

Importing this package registers the built-in providers. Add a provider by
creating a module with a ``PROVIDERS.register(...)`` call and importing it here.
"""

from pr_review.providers import gemini, mock, openai_compat  # noqa: F401
from pr_review.providers.base import PROVIDERS, LLMProvider

__all__ = ["PROVIDERS", "LLMProvider"]
