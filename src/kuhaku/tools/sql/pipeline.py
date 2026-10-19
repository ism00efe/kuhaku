"""Composable text-to-SQL pipeline: named steps that can be added, removed, replaced,
enabled or disabled -- the extraction tool's model, with one difference.

Two steps are the security boundary and are **protected**: ``schema`` (what the model
may see) and ``validate`` (what may run). They cannot be removed, disabled or
replaced. ``generate`` cannot be removed or disabled, but can be replaced by another
step of the same name -- whatever it produces still goes through ``validate``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from kuhaku.core.exceptions import ConfigError
from kuhaku.core.observability import instrumented_step

from .context import SQLContext

SCHEMA_STEP = "schema"
GENERATE_STEP = "generate"
VALIDATE_STEP = "validate"
EXECUTE_STEP = "execute"
ANSWER_STEP = "answer"

_REQUIRED = (SCHEMA_STEP, GENERATE_STEP, VALIDATE_STEP)
_PROTECTED = frozenset({SCHEMA_STEP, VALIDATE_STEP})

logger = logging.getLogger(__name__)


class Step:
    """Base class for a pipeline stage."""

    name: str = ""

    def run(self, ctx: SQLContext) -> None:
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
        missing = [name for name in _REQUIRED if name not in self.names]
        if missing:
            raise ConfigError(f"A text-to-SQL pipeline needs the steps {missing}.")
        order = [self.names.index(name) for name in _REQUIRED]
        if order != sorted(order):
            raise ConfigError(f"The steps {list(_REQUIRED)} must run in that order.")

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

    def add(self, step: Step, *, before: str | None = None, after: str | None = None) -> None:
        if before is not None and after is not None:
            raise ConfigError("Pass either 'before' or 'after', not both.")
        self._check_new(step)
        if before is not None:
            if before == SCHEMA_STEP:
                raise ConfigError(f"Nothing may run before the '{SCHEMA_STEP}' step.")
            index = self._index(before)
        elif after is not None:
            index = self._index(after) + 1
        else:
            index = len(self._entries)
        self._entries.insert(index, _Entry(step))

    def remove(self, name: str) -> Step:
        self._refuse(name, "remove")
        return self._entries.pop(self._index(name)).step

    def replace(self, name: str, step: Step) -> Step:
        if name in _PROTECTED:
            raise ConfigError(
                f"The '{name}' step is part of the access-control boundary and cannot be "
                f"replaced."
            )
        if step.name != name:
            raise ConfigError(
                f"A replacement for step '{name}' must have the same name; got '{step.name}'."
            )
        entry = self._entry(name)
        old, entry.step = entry.step, step
        return old

    def enable(self, name: str) -> None:
        self._entry(name).enabled = True

    def disable(self, name: str) -> None:
        self._refuse(name, "disable")
        self._entry(name).enabled = False

    def run(self, ctx: SQLContext) -> None:
        for entry in self._entries:
            if not entry.enabled:
                continue
            if ctx.stopped:
                break
            with instrumented_step(f"sql.{entry.step.name}", logger) as rec:
                entry.step.run(ctx)
                rec.set(issue_count=len(ctx.issues))

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

    def _refuse(self, name: str, action: str) -> None:
        if name in _PROTECTED:
            raise ConfigError(
                f"Cannot {action} the '{name}' step -- it is part of the access-control "
                f"boundary."
            )
        if name == GENERATE_STEP:
            raise ConfigError(
                f"Cannot {action} the '{GENERATE_STEP}' step. Use "
                f"replace('{GENERATE_STEP}', ...) to change how SQL is produced."
            )

    def _index(self, name: str) -> int:
        for index, entry in enumerate(self._entries):
            if entry.step.name == name:
                return index
        raise ConfigError(f"No step named '{name}'. Steps: {self.names}.")

    def _entry(self, name: str) -> _Entry:
        return self._entries[self._index(name)]
