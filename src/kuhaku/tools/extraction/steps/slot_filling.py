"""Multi-turn extraction: carry values from earlier turns into this one."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ValidationError

from ..context import ExtractionContext
from ..parsing import SCHEMA_ISSUE_CODES, validate_data
from ..pipeline import Step
from ..prompts import render

SLOT_FILLING_STEP = "slot_filling"


class SlotFillingStep(Step):
    """Merges this turn's extraction into the data an earlier turn left behind.

    Given ``extract(..., prior=earlier_result)``, the model is shown what is already
    known, and afterwards the two are merged: a non-null value from this turn wins,
    a null keeps the earlier value. So "5 kilo" after "çay yaz" completes the earlier
    extraction instead of starting a new one with no product.

    The tool keeps no state between calls -- the caller passes the previous result
    back in, the same way the conversation history is passed. Inert when no ``prior``
    is given.

    Values in ``prior`` that break the schema (a failed earlier turn keeps its data) are
    dropped to null first, so a bad value is neither shown to the model as "already
    known" nor merged into this turn.

    Known limit: because null means "keep", a later turn cannot clear a value back to
    null through the merge.
    """

    name = SLOT_FILLING_STEP

    def prepare(self, ctx: ExtractionContext) -> None:
        if ctx.prior_data:
            ctx.prior_data = drop_invalid(ctx.schema.model, ctx.prior_data)
            ctx.prompt_blocks.append((ctx.prompts.slot_filling, render(ctx.prior_data)))

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


def drop_invalid(model: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``data`` with every value that breaks ``model`` set to null. Missing
    values are left alone -- an incomplete earlier turn is what slot filling is for."""

    try:
        model.model_validate(data)
        return data
    except ValidationError as exc:
        errors = [error for error in exc.errors() if error["type"] != "missing"]
    cleaned = copy.deepcopy(data)
    for error in errors:
        *parents, leaf = error["loc"] or (None,)
        node: Any = cleaned
        for part in parents:
            try:
                node = node[part]
            except (KeyError, IndexError, TypeError):
                node = None
                break
        in_dict = isinstance(node, dict) and leaf in node
        in_list = isinstance(node, list) and isinstance(leaf, int) and leaf < len(node)
        if in_dict or in_list:
            node[leaf] = None
    return cleaned
