"""Fakes for the capability-resolution acceptance suite (spec §15).

The mechanism under test (``kuhaku.core.resolve``) is tool-agnostic and carries no
knowledge of Ollama/Chroma/etc., so every adapter here is a scriptable fake. The real
``Registry`` is used with these fakes -- only the adapters, UI and (for the branch-count
checks) the memory are doubled.

Until the mechanism lands (commit 3), the whole directory skips rather than errors --
``pytest`` here runs without ``--strict-markers`` or fail-on-skip, so this keeps commit 1
green in CI while the suite is written first.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

try:  # the mechanism lands in commit 3; until then, ignore this directory cleanly
    from kuhaku.core.resolve import Candidate, Cost, Environment
except ImportError:  # pragma: no cover - collection-time guard
    collect_ignore_glob = ["test_*.py"]
    Candidate = Cost = Environment = object  # placeholders so this conftest still imports


def make_env(**overrides) -> Environment:
    base = dict(
        python="3.12",
        in_isolated_python=True,
        isolation_source="venv",
        gpu=None,
        vram_class="unknown",
        os="linux",
    )
    base.update(overrides)
    return Environment(**base)


def make_candidate(
    candidate_id: str,
    kind: str = "llm",
    *,
    ready: bool = True,
    safety_rank: int = 0,
    label: str | None = None,
    cost: Cost | None = None,
    activate=None,
) -> Candidate:
    return Candidate(
        id=candidate_id,
        kind=kind,
        label=label if label is not None else f"{candidate_id} ({kind})",
        cost=cost if cost is not None else Cost(note=f"{candidate_id} note"),
        ready=ready,
        safety_rank=safety_rank,
        activate=activate if activate is not None else (lambda: candidate_id),
    )


class FakeAdapter:
    """One ``kind``, a fixed candidate list, an optional documented baseline, and the
    set of packages the decision depends on (amendment 5)."""

    def __init__(
        self,
        kind: str,
        candidates: Sequence[Candidate],
        *,
        baseline_id: str | None = None,
        packages: frozenset[str] = frozenset(),
    ) -> None:
        self.kind = kind
        self.packages = packages
        self._candidates = list(candidates)
        self._baseline_id = baseline_id
        self.probe_calls = 0

    def probe(self, env: Environment) -> Sequence[Candidate]:
        self.probe_calls += 1
        return list(self._candidates)

    def baseline(self, env: Environment) -> Candidate | None:
        return next((c for c in self._candidates if c.id == self._baseline_id), None)


class FakeUI:
    def __init__(
        self,
        *,
        interactive: bool = False,
        ask_returns=None,
        confirm_map: dict[str, bool] | None = None,
        confirm_default: bool = False,
    ) -> None:
        self._interactive = interactive
        self._ask_returns = ask_returns
        self._confirm_map = confirm_map or {}
        self._confirm_default = confirm_default
        self.announcements: list[tuple[str, bool, bool]] = []
        self.ask_calls: list[tuple[str, list[Candidate]]] = []
        self.confirm_calls: list[tuple[str, Cost]] = []

    # -- port -----------------------------------------------------------------
    def is_interactive(self) -> bool:
        return self._interactive

    def announce(self, message, *, prominent=False, degraded=False, dedupe_key=None):
        self.announcements.append((message, prominent, degraded))

    def ask(self, question, options):
        options = list(options)
        self.ask_calls.append((question, options))
        r = self._ask_returns
        if callable(r):
            return r(options)
        if isinstance(r, str):
            return next((o for o in options if o.id == r), None)
        return r

    def confirm(self, action, cost):
        self.confirm_calls.append((action, cost))
        for needle, verdict in self._confirm_map.items():
            if needle in action:
                return verdict
        return self._confirm_default

    # -- helpers ------------------------------------------------------------
    @property
    def messages(self) -> list[str]:
        return [m for m, _, _ in self.announcements]

    @property
    def prominent_messages(self) -> list[str]:
        return [m for m, prominent, _ in self.announcements if prominent]

    @property
    def confirm_actions(self) -> list[str]:
        return [a for a, _ in self.confirm_calls]


class FakeMemory:
    """Ignores the fingerprint -- for the branch-count behavioural checks only.
    Checks 6 and 12 use the real ``JsonMemory``."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}
        self.get_calls: list[str] = []
        self.put_calls: list[tuple[str, str]] = []

    def get(self, kind, fp):
        self.get_calls.append(kind)
        return self._d.get(kind)

    def put(self, kind, fp, candidate_id):
        self.put_calls.append((kind, candidate_id))
        self._d[kind] = candidate_id

    def reset(self, kind=None):
        if kind is None:
            self._d.clear()
        else:
            self._d.pop(kind, None)


@pytest.fixture
def env() -> Environment:
    return make_env()


@pytest.fixture
def memory() -> FakeMemory:
    return FakeMemory()
