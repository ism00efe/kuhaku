"""Optional step: the caller's own checks on a decision, able to veto it."""

from __future__ import annotations

from collections.abc import Callable

from kuhaku.core.exceptions import CustomComponentError

from ..context import ClassificationContext
from ..pipeline import Step
from ..result import UndecidedReason

Validator = Callable[[tuple[str, ...], ClassificationContext], "str | None"]
"""``(labels, ctx) -> None to accept, or the reason for rejecting``."""


class ValidationStep(Step):
    """Runs domain checks a label set cannot express -- "``refund`` only with an order
    number", "``urgent`` and ``spam`` never together". A rejection turns the result
    into ``UNDECIDED`` with reason ``REJECTED`` rather than editing the labels: which
    label a vetoed text should get instead is not this step's to guess.

    Runs only on a decision, from whichever step made it.
    """

    name = "validate"

    def __init__(self, *validators: Validator) -> None:
        for validator in validators:
            if not callable(validator):
                raise CustomComponentError(f"Validator {validator!r} is not callable.")
        self.validators = list(validators)

    def run(self, ctx: ClassificationContext) -> None:
        for validator in self.validators:
            if not ctx.chosen:
                return
            rejection = validator(ctx.chosen, ctx)
            if rejection is None:
                continue
            if not isinstance(rejection, str) or not rejection:
                raise CustomComponentError(
                    f"Validator {validator!r} returned {rejection!r}; expected None or a "
                    f"non-empty reason string."
                )
            ctx.abstain(UndecidedReason.REJECTED, by=self.name, reason=rejection)
