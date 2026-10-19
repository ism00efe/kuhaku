"""The generic step pipeline under classification's :mod:`.pipeline`.

An ordered list of named :class:`Step` objects that can be added, removed, replaced,
enabled or disabled, run against one mutable context object that the steps communicate
through.

A run has two phases. ``prepare`` runs for every enabled step first -- a step placed
after the core one can still contribute something (a prompt rule, a response key) before
the core step runs. ``run`` then executes the enabled steps in order and stops early
once the context's ``stopped`` attribute is true.

Policy is declared per pipeline, not hard-coded:

- ``required`` steps must be present; they cannot be removed or disabled, but can be
  replaced by another step of the same name. The mapping's value says *why* (it ends up
  in the refusal message).
- ``protected`` steps (a subset of what must be present) cannot be removed, disabled or
  replaced at all.
- ``ordered=True`` makes the required steps keep their relative order as given.
- ``first`` names a step nothing may be inserted before.
- ``metric_prefix`` prefixes each step's span/metric name (``"<prefix>.<step>"``).
- :meth:`Pipeline.step_fields` (override it) adds per-step span fields after each step.

"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from kuhaku.core.exceptions import ConfigError
from kuhaku.core.observability import instrumented_step

C = TypeVar("C")

logger = logging.getLogger(__name__)

__all__ = ["Pipeline", "Step"]


class Step(Generic[C]):
    """Base class for a pipeline stage. Override either hook, or both."""

    name: str = ""

    def prepare(self, ctx: C) -> None:
        """Called for every enabled step before any step runs."""

    def run(self, ctx: C) -> None:
        """Do this step's work, reading and writing only ``ctx``."""


@dataclass
class _Entry(Generic[C]):
    step: Step[C]
    enabled: bool = True


class Pipeline(Generic[C]):
    """An ordered list of named steps. See the module docstring for the policy knobs.

    ``label`` names the pipeline in error messages ("An extraction pipeline needs ...");
    ``log`` is the logger the per-step records go to (the tool's own, so a caller
    filtering on the tool's logger sees them).
    """

    def __init__(
        self,
        steps: Iterable[Step[C]],
        *,
        required: Mapping[str, str] | Sequence[str] = (),
        protected: Mapping[str, str] | Sequence[str] = (),
        ordered: bool = False,
        first: str | None = None,
        metric_prefix: str = "pipeline",
        label: str = "",
        log: logging.Logger | None = None,
    ) -> None:
        self._required = _reasons(required)
        self._protected = _reasons(protected)
        self._ordered = ordered
        self._first = first
        self.metric_prefix = metric_prefix
        self._label = label or metric_prefix
        self._log = log or logger
        self._entries: list[_Entry[C]] = []
        for step in steps:
            self._check_new(step)
            self._entries.append(_Entry(step))
        needed = [*self._required, *(n for n in self._protected if n not in self._required)]
        missing = [name for name in needed if name not in self.names]
        if missing:
            listed = ", ".join(f"'{name}'" for name in missing)
            raise ConfigError(
                f"A{'n' if self._label[:1] in 'aeiou' else ''} {self._label} pipeline needs "
                f"a step named {listed}; got {self.names or 'no steps'}. Include it (or a "
                f"replacement with the same name)."
            )
        if ordered:
            positions = [self.names.index(name) for name in self._required]
            if positions != sorted(positions):
                raise ConfigError(
                    f"The steps {list(self._required)} must run in that order; got {self.names}."
                )
        if first is not None and self.names and self.names[0] != first:
            raise ConfigError(f"The '{first}' step must run first; got {self.names}.")

    # --- inspection -------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [entry.step.name for entry in self._entries]

    def get(self, name: str) -> Step[C]:
        return self._entry(name).step

    def is_enabled(self, name: str) -> bool:
        return self._entry(name).enabled

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __repr__(self) -> str:
        parts = [e.step.name + ("" if e.enabled else " (disabled)") for e in self._entries]
        return f"{type(self).__name__}([{', '.join(parts)}])"

    # --- composition ------------------------------------------------------------------

    def add(self, step: Step[C], *, before: str | None = None, after: str | None = None) -> None:
        """Insert ``step``; at the end unless ``before`` or ``after`` names an existing step."""

        if before is not None and after is not None:
            raise ConfigError("Pass either 'before' or 'after', not both.")
        self._check_new(step)
        if before is not None:
            if before == self._first:
                raise ConfigError(f"Nothing may run before the '{before}' step.")
            index = self._index(before)
        elif after is not None:
            index = self._index(after) + 1
        else:
            index = len(self._entries)
        self._entries.insert(index, _Entry(step))

    def remove(self, name: str) -> Step[C]:
        self._refuse(name, "remove")
        return self._entries.pop(self._index(name)).step

    def replace(self, name: str, step: Step[C]) -> Step[C]:
        """Swap the implementation of an existing step, keeping its position and state."""

        if name in self._protected:
            raise ConfigError(
                f"The '{name}' step cannot be replaced -- {self._protected[name]}."
            )
        if step.name != name:
            raise ConfigError(
                f"A replacement for step '{name}' must have the same name; got '{step.name}'."
            )
        if not isinstance(step, Step):
            raise ConfigError(f"Pipeline steps must subclass Step; got {type(step).__name__}.")
        entry = self._entry(name)
        old, entry.step = entry.step, step
        return old

    def enable(self, name: str) -> None:
        self._entry(name).enabled = True

    def disable(self, name: str) -> None:
        self._refuse(name, "disable")
        self._entry(name).enabled = False

    # --- execution --------------------------------------------------------------------

    def run(self, ctx: C) -> None:
        active = [entry.step for entry in self._entries if entry.enabled]
        for step in active:
            step.prepare(ctx)
        for step in active:
            if getattr(ctx, "stopped", False):
                break
            with instrumented_step(f"{self.metric_prefix}.{step.name}", self._log) as rec:
                step.run(ctx)
                fields = self.step_fields(ctx)
                if fields:
                    rec.set(**fields)

    def step_fields(self, ctx: C) -> dict[str, Any]:
        """Span/log fields recorded after each step. Override per tool."""

        return {}

    # --- internals --------------------------------------------------------------------

    def _check_new(self, step: Step[C]) -> None:
        if not isinstance(step, Step):
            raise ConfigError(f"Pipeline steps must subclass Step; got {type(step).__name__}.")
        if not step.name:
            raise ConfigError(f"{type(step).__name__} has no name; set a class-level 'name'.")
        if step.name in self.names:
            raise ConfigError(
                f"A step named '{step.name}' is already in the pipeline; "
                f"use replace('{step.name}', ...) to swap it."
            )

    def _refuse(self, name: str, action: str) -> None:
        if name in self._protected:
            raise ConfigError(f"Cannot {action} the '{name}' step -- {self._protected[name]}.")
        if name in self._required:
            raise ConfigError(
                f"Cannot {action} the '{name}' step -- {self._required[name]}. "
                f"Use replace('{name}', ...) to change how it works."
            )

    def _index(self, name: str) -> int:
        for index, entry in enumerate(self._entries):
            if entry.step.name == name:
                return index
        raise ConfigError(f"No step named '{name}'. Steps: {self.names}.")

    def _entry(self, name: str) -> _Entry[C]:
        return self._entries[self._index(name)]


def _reasons(names: Mapping[str, str] | Sequence[str]) -> dict[str, str]:
    if isinstance(names, Mapping):
        return dict(names)
    if isinstance(names, str):
        return {names: "it is required"}
    return {name: "it is required" for name in names}
