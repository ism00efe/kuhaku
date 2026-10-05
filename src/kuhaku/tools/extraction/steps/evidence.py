"""Optional step: make the model quote where each value came from, and check the quote."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from typing import Any

from kuhaku.core.exceptions import ConfigError

from ..context import ExtractionContext
from ..parsing import json_pointer
from ..pipeline import Step
from ..result import FieldReport, Issue, IssueKind, Provenance

_WHITESPACE = re.compile(r"\s+")
_ON_UNGROUNDED = ("report", "invalid")


class EvidenceStep(Step):
    """Records, per extracted value, whether its quoted evidence appears in the source.

    The check is deterministic -- a normalized substring match against the source
    text -- because asking the model whether its own quote is real would only repeat
    the claim being checked.

    ``on_ungrounded="report"`` (default) only fills ``result.fields``.
    ``on_ungrounded="invalid"`` additionally turns every ungrounded value into an
    ``INVALID`` issue, so a result is ``COMPLETE`` only when all values are grounded.
    """

    name = "evidence"

    def __init__(self, *, on_ungrounded: str = "report") -> None:
        if on_ungrounded not in _ON_UNGROUNDED:
            raise ConfigError(
                f"on_ungrounded must be one of {_ON_UNGROUNDED}; got '{on_ungrounded}'."
            )
        self.on_ungrounded = on_ungrounded

    def prepare(self, ctx: ExtractionContext) -> None:
        ctx.instructions.append(
            'For every non-null value in "data", add an entry to "evidence": the key is '
            "the value's JSON Pointer (for example /items/0/name) and the value is the "
            "passage of the source text it came from, copied exactly."
        )
        ctx.response_fields["evidence"] = (
            "an object mapping each JSON Pointer to a passage copied exactly from the source text"
        )

    def run(self, ctx: ExtractionContext) -> None:
        if ctx.data is None:
            return
        raw_evidence = (ctx.response or {}).get("evidence")
        evidence = raw_evidence if isinstance(raw_evidence, dict) else {}
        source = _normalize(ctx.text)

        for loc, value in _leaves(ctx.data):
            path = json_pointer(loc)
            if value is None:
                ctx.fields[path] = FieldReport(path, Provenance.MISSING)
                continue
            quote = evidence.get(path)
            quote = quote if isinstance(quote, str) and quote.strip() else None
            if quote is not None and _normalize(quote) in source:
                ctx.fields[path] = FieldReport(path, Provenance.SOURCE, quote)
                continue
            if ctx.prior_data is not None and _lookup(ctx.prior_data, loc) == value:
                ctx.fields[path] = FieldReport(path, Provenance.PRIOR)
                continue
            ctx.fields[path] = FieldReport(path, Provenance.UNGROUNDED, quote)
            if self.on_ungrounded == "invalid":
                ctx.issues.append(
                    Issue(
                        IssueKind.INVALID,
                        "ungrounded",
                        f"{path}: no supporting passage found in the source text",
                        path,
                    )
                )


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _leaves(node: Any, loc: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _leaves(child, (*loc, key))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from _leaves(child, (*loc, index))
    else:
        yield loc, node


_ABSENT = object()


def _lookup(node: Any, loc: tuple[Any, ...]) -> Any:
    for part in loc:
        in_dict = isinstance(node, dict) and part in node
        in_list = isinstance(node, list) and isinstance(part, int) and part < len(node)
        if not (in_dict or in_list):
            return _ABSENT
        node = node[part]
    return node
