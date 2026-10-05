"""Composable extraction pipeline.

Every stage around the core extraction -- evidence checking, caller validators, and
later sanitization, the input guard, routing, audit -- is a :class:`Step`. A pipeline is
an ordered list of named steps that can be added, removed, replaced, enabled or
disabled. The one step that cannot be removed or disabled is ``extract``: without it
there is nothing to check. It can still be *replaced* by another step of the same name.

A run has two phases. ``prepare`` runs for every enabled step first, so a step placed
after ``extract`` (evidence, for instance) can still ask the model for something before
the model is called. ``run`` then executes in order and stops early once a step calls
:meth:`ExtractionContext.stop`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from kuhaku.core.exceptions import ConfigError

from .context import ExtractionContext

EXTRACT_STEP = "extract"


class Step:
    """Base class for a pipeline stage. Override either hook, or both."""

    name: str = ""

    def prepare(self, ctx: ExtractionContext) -> None:
        """Called before any step runs -- contribute prompt rules or response keys."""

    def run(self, ctx: ExtractionContext) -> None:
        """Do this step's work, reading and writing only ``ctx``."""


@dataclass
class _Entry:
    step: Step
    enabled: bool = True


class Pipeline:
    def __init__(self, steps: Iterable[Step]) -> None:
        self._entries: list[_Entry] = []
        for step in steps:
            self._check_new(step)
            self._entries.append(_Entry(step))
        if EXTRACT_STEP not in self.names:
            raise ConfigError(
                f"An extraction pipeline needs a step named '{EXTRACT_STEP}'; "
                f"got {self.names or 'no steps'}. Include ExtractStep (or a replacement "
                f"with name='{EXTRACT_STEP}')."
            )

    # --- inspection -------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [entry.step.name for entry in self._entries]

    def get(self, name: str) -> Step:
        return self._entry(name).step

    def is_enabled(self, name: str) -> bool:
        return self._entry(name).enabled

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __repr__(self) -> str:
        parts = [e.step.name + ("" if e.enabled else " (disabled)") for e in self._entries]
        return f"Pipeline([{', '.join(parts)}])"

    # --- composition ------------------------------------------------------------------

    def add(self, step: Step, *, before: str | None = None, after: str | None = None) -> None:
        """Insert ``step``; at the end unless ``before`` or ``after`` names an existing step."""

        if before is not None and after is not None:
            raise ConfigError("Pass either 'before' or 'after', not both.")
        self._check_new(step)
        if before is not None:
            index = self._index(before)
        elif after is not None:
            index = self._index(after) + 1
        else:
            index = len(self._entries)
        self._entries.insert(index, _Entry(step))

    def remove(self, name: str) -> Step:
        self._refuse_if_core(name, "remove")
        return self._entries.pop(self._index(name)).step

    def replace(self, name: str, step: Step) -> Step:
        """Swap the implementation of an existing step, keeping its position and state."""

        if step.name != name:
            raise ConfigError(
                f"A replacement for step '{name}' must have the same name; "
                f"got '{step.name}'."
            )
        entry = self._entry(name)
        old, entry.step = entry.step, step
        return old

    def enable(self, name: str) -> None:
        self._entry(name).enabled = True

    def disable(self, name: str) -> None:
        self._refuse_if_core(name, "disable")
        self._entry(name).enabled = False

    # --- execution --------------------------------------------------------------------

    def run(self, ctx: ExtractionContext) -> None:
        active = [entry.step for entry in self._entries if entry.enabled]
        for step in active:
            step.prepare(ctx)
        for step in active:
            if ctx.stopped:
                break
            step.run(ctx)

    # --- internals --------------------------------------------------------------------

    def _check_new(self, step: Step) -> None:
        if not isinstance(step, Step):
            raise ConfigError(f"Pipeline steps must subclass Step; got {type(step).__name__}.")
        if not step.name:
            raise ConfigError(f"{type(step).__name__} has no name; set a class-level 'name'.")
        if step.name in self.names:
            raise ConfigError(
                f"A step named '{step.name}' is already in the pipeline; "
                f"use replace('{step.name}', ...) to swap it."
            )

    def _refuse_if_core(self, name: str, action: str) -> None:
        if name == EXTRACT_STEP:
            raise ConfigError(
                f"Cannot {action} the '{EXTRACT_STEP}' step -- it is the extraction itself. "
                f"Use replace('{EXTRACT_STEP}', ...) to change how it works."
            )

    def _index(self, name: str) -> int:
        for index, entry in enumerate(self._entries):
            if entry.step.name == name:
                return index
        raise ConfigError(f"No step named '{name}'. Steps: {self.names}.")

    def _entry(self, name: str) -> _Entry:
        return self._entries[self._index(name)]
