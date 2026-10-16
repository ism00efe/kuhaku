"""Structured-output metric: field-by-field comparison of a JSON answer to a JSON label.

Tool-agnostic: it compares two JSON objects and knows nothing about how the answer was
produced. A target that answers with a JSON object (the structured extraction tool's
``evaluate_sample`` does) is scored against a golden dataset whose ``golden_answer`` is
that object serialized as JSON, so the existing JSONL schema needs no new field.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..base_metric import BaseMetric
from ..sample import EvaluationSample


class StructuredMatchMetric(BaseMetric):
    """Scores ``sample.answer`` (a JSON object) against ``metadata["golden_answer"]``.

    - ``field_accuracy``: share of the label's leaf values (by JSON Pointer) the answer
      has equal. A label value of null matches an absent or null answer value.
    - ``exact_match``: 1.0 when every label leaf matches, else 0.0.

    Answer keys the label does not have are not counted -- the label says what matters.
    Strings compare after stripping and case-folding unless ``normalize_strings=False``;
    numbers compare by value (5 == 5.0).

    Not applicable (``{}``) when the label is not a JSON object -- a dataset written for
    another kind of target. An answer that is missing or not a JSON object scores 0.
    """

    name = "structured_match"

    def __init__(self, *, normalize_strings: bool = True) -> None:
        self.normalize_strings = normalize_strings

    def evaluate(self, sample: EvaluationSample) -> dict[str, float]:
        golden = _as_object(sample.metadata.get("golden_answer"))
        if golden is None:
            return {}
        answer = _as_object(sample.answer) or {}

        expected = dict(_leaves(golden))
        if not expected:
            return {}
        got = dict(_leaves(answer))
        matched = sum(
            1 for path, value in expected.items() if self._equal(value, got.get(path))
        )
        return {
            "field_accuracy": matched / len(expected),
            "exact_match": 1.0 if matched == len(expected) else 0.0,
        }

    def _equal(self, expected: Any, got: Any) -> bool:
        if self.normalize_strings and isinstance(expected, str) and isinstance(got, str):
            return expected.strip().casefold() == got.strip().casefold()
        if isinstance(expected, bool) or isinstance(got, bool):
            return expected is got
        return bool(expected == got)


def _as_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _leaves(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict) and node:
        for key, child in node.items():
            yield from _leaves(child, f"{path}/{str(key).replace('~', '~0').replace('/', '~1')}")
    elif isinstance(node, list) and node:
        for index, child in enumerate(node):
            yield from _leaves(child, f"{path}/{index}")
    else:
        yield path, node
