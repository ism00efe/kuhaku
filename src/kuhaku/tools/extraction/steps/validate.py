"""Optional step: the caller's own rules, run on the validated model instance."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from kuhaku.core.exceptions import CustomComponentError

from ..context import ExtractionContext
from ..pipeline import Step
from ..result import Issue, IssueKind

Validator = Callable[[Any, ExtractionContext], Iterable[Issue] | None]
"""``(value, ctx) -> issues``. ``value`` is the validated schema model instance."""

# REPAIRABLE is excluded: there is no loop back to the model after this step yet, so a
# validator returning it would silently turn into FAILED instead of a repair.
_ALLOWED_KINDS = (IssueKind.NEEDS_INPUT, IssueKind.INVALID)


class ValidationStep(Step):
    """Runs domain checks the schema cannot express -- a catalog lookup, a limit, a
    cross-field rule. Runs only when the model's output passed schema validation."""

    name = "validate"

    def __init__(self, *validators: Validator) -> None:
        for validator in validators:
            if not callable(validator):
                raise CustomComponentError(f"Validator {validator!r} is not callable.")
        self.validators = list(validators)

    def run(self, ctx: ExtractionContext) -> None:
        value: BaseModel | None = ctx.value
        if value is None:
            return
        for validator in self.validators:
            for issue in validator(value, ctx) or ():
                if not isinstance(issue, Issue) or issue.kind not in _ALLOWED_KINDS:
                    raise CustomComponentError(
                        f"Validator {validator!r} returned {issue!r}; expected an Issue of "
                        f"kind {[kind.value for kind in _ALLOWED_KINDS]}."
                    )
                ctx.issues.append(issue)
