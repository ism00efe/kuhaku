"""Multi-turn extraction: carry values from earlier turns into this one."""

from __future__ import annotations

from typing import Any

from ..context import ExtractionContext
from ..parsing import SCHEMA_ISSUE_CODES, validate_data
from ..pipeline import Step
from ..prompts import render

CONTINUATION_STEP = "continuation"


class ContinuationStep(Step):
    """Merges this turn's extraction into the data an earlier turn left behind.

    Given ``extract(..., prior=earlier_result)``, the model is shown what is already
    known, and afterwards the two are merged: a non-null value from this turn wins,
    a null keeps the earlier value. So "5 kilo" after "çay yaz" completes the earlier
    extraction instead of starting a new one with no product.

    The tool keeps no state between calls -- the caller passes the previous result
    back in, the same way the conversation history is passed. Inert when no ``prior``
    is given.

    Known limit: because null means "keep", a later turn cannot clear a value back to
    null through the merge.
    """

    name = CONTINUATION_STEP

    def prepare(self, ctx: ExtractionContext) -> None:
        if ctx.prior_data:
            title = (
                "Values already extracted in earlier turns (repeat or change them only "
                "if the source text does; use null for anything the source text does "
                "not mention)"
            )
            ctx.prompt_blocks.append((title, render(ctx.prior_data)))

    def run(self, ctx: ExtractionContext) -> None:
        if not ctx.prior_data or ctx.data is None:
            return
        ctx.data = merge(ctx.prior_data, ctx.data)
        ctx.issues[:] = [issue for issue in ctx.issues if issue.code not in SCHEMA_ISSUE_CODES]
        value, missing, errors = validate_data(ctx.schema.model, ctx.data)
        ctx.value = value
        ctx.issues.extend(errors + missing)


def merge(earlier: Any, current: Any) -> Any:
    """Deep-merge ``current`` over ``earlier``: dicts key by key, null keeps ``earlier``,
    anything else (a scalar, a non-empty list) replaces it."""

    if current is None:
        return earlier
    if isinstance(earlier, dict) and isinstance(current, dict):
        merged = dict(earlier)
        for key, value in current.items():
            merged[key] = merge(earlier.get(key), value)
        return merged
    if isinstance(current, list) and not current:
        return earlier if earlier is not None else current
    return current
