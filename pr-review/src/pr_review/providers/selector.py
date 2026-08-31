"""Resolve a logical model tier to a concrete, instantiated provider + model.

Centralises the "which model runs this" decision so no other module hard-codes a
provider name, and caches one transport instance per provider for the process.
Ordering and failover across a tier's candidates belong to
:mod:`pr_review.providers.dispatch`; this module only builds what that chain
names.
"""

from __future__ import annotations

from dataclasses import dataclass

from pr_review.config import Candidate, Config
from pr_review.providers.base import LLMProvider, build


@dataclass(frozen=True)
class ResolvedModel:
    provider: LLMProvider
    provider_name: str
    model: str
    max_tokens: int
    temperature: float


class ModelSelector:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._cache: dict[str, LLMProvider] = {}

    def _provider(self, name: str) -> LLMProvider:
        if name not in self._cache:
            pc = self.config.provider(name)
            self._cache[name] = build(pc.kind, pc)
        return self._cache[name]

    def provider_for(self, candidate: Candidate) -> LLMProvider:
        """Instantiated transport for one :class:`Candidate`."""
        return self._provider(candidate.provider)

    def for_tier(self, tier_name: str) -> ResolvedModel:
        tier = self.config.tier(tier_name)
        provider_name, model = self.config.resolve_provider(tier_name)
        return ResolvedModel(
            provider=self._provider(provider_name),
            provider_name=provider_name,
            model=model,
            max_tokens=tier.max_tokens,
            temperature=tier.temperature,
        )
