"""The adapter registry. The resolver asks it for a ``kind`` and never enumerates
adapters by name."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .cost import Candidate
from .environment import Environment

_log = logging.getLogger("kuhaku")


@runtime_checkable
class Adapter(Protocol):
    kind: str
    """The decision kind this adapter contributes to."""
    packages: frozenset[str]
    """Distribution names this adapter's decision depends on -- unioned across the
    registry into the fingerprint's ``packages`` field, so installing or removing one
    re-opens exactly the decisions that could change."""

    def probe(self, env: Environment) -> Sequence[Candidate]:
        """The candidates usable *right now*, with no install and no download. Cheap.
        Never installs, never downloads, never makes an outbound network call to decide
        whether a local resource is available, and never raises -- on any internal
        failure it returns an empty sequence."""

    def baseline(self, env: Environment) -> Candidate | None:
        """The documented-baseline candidate, constructed without probing. Used only when
        ``KUHAKU_AUTO`` is disabled. ``None`` if this adapter has no baseline."""


class Registry:
    def __init__(self) -> None:
        self._adapters: list[Adapter] = []

    def register(self, adapter: Adapter) -> None:
        self._adapters.append(adapter)

    def adapters_for(self, kind: str) -> list[Adapter]:
        return [a for a in self._adapters if a.kind == kind]

    def candidates(self, kind: str, env: Environment) -> list[Candidate]:
        out: list[Candidate] = []
        for adapter in self.adapters_for(kind):
            try:
                out.extend(adapter.probe(env))
            except Exception:
                # probe() is contracted never to raise. If one does it is a bug in that
                # adapter, not a reason to fail the whole kind -- but log it, so a
                # misconfigured adapter is not an invisible "zero candidates".
                _log.warning(
                    "adapter %r raised in probe(%s); treating as no candidates",
                    type(adapter).__name__, kind, exc_info=True,
                )
                continue
        return out

    def baseline(self, kind: str, env: Environment) -> Candidate | None:
        for adapter in self.adapters_for(kind):
            try:
                found = adapter.baseline(env)
            except Exception:
                found = None
            if found is not None:
                return found
        return None

    def required_packages(self) -> frozenset[str]:
        result: frozenset[str] = frozenset()
        for adapter in self._adapters:
            result |= getattr(adapter, "packages", frozenset())
        return result
