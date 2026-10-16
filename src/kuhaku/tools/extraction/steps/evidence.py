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
    the claim being checked. A real quote must also contain the value it is cited for
    (a string as written, a number in its usual digit forms), otherwise the value is
    ``MISMATCH``; booleans are not checked against the quote, they are never written
    out. A value the schema expects in a form the text does not use (a date written
    "tomorrow") is reported as ``MISMATCH`` too -- it is not stated as such.

    ``on_ungrounded="report"`` (default) only fills ``result.fields``.
    ``on_ungrounded="invalid"`` additionally turns every ungrounded or mismatched value
    into an ``INVALID`` issue, so a result is ``COMPLETE`` only when all values are
    grounded.
    """

    name = "evidence"

    def __init__(self, *, on_ungrounded: str = "report") -> None:
        if on_ungrounded not in _ON_UNGROUNDED:
            raise ConfigError(
                f"on_ungrounded must be one of {_ON_UNGROUNDED}; got '{on_ungrounded}'."
            )
        self.on_ungrounded = on_ungrounded

    def prepare(self, ctx: ExtractionContext) -> None:
        ctx.instructions.append(ctx.prompts.evidence)
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
            quoted = quote is not None and _normalize(quote) in source
            if quote is not None and quoted and _states(quote, value):
                ctx.fields[path] = FieldReport(path, Provenance.SOURCE, quote)
                continue
            if ctx.prior_data is not None and _lookup(ctx.prior_data, loc) == value:
                ctx.fields[path] = FieldReport(path, Provenance.PRIOR)
                continue
            provenance = Provenance.MISMATCH if quoted else Provenance.UNGROUNDED
            ctx.fields[path] = FieldReport(path, provenance, quote)
            if self.on_ungrounded == "invalid":
                problem = (
                    "the quoted passage does not contain the value"
                    if quoted
                    else "no supporting passage found in the source text"
                )
                ctx.issues.append(
                    Issue(IssueKind.INVALID, provenance.value, f"{path}: {problem}", path)
                )


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    # casefold() turns "İ" into "i" + a combining dot and leaves "ı" alone, so Turkish
    # text written in capitals never matched its lower-case form. All four i's fold to
    # one letter; matching only needs both sides folded the same way.
    folded = folded.replace("\u0307", "").replace("\u0131", "i")
    return _WHITESPACE.sub(" ", folded).strip()


def _states(quote: str, value: Any) -> bool:
    """Whether ``quote`` contains ``value`` as written."""

    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        text = _normalize(quote)
        return any(
            re.search(rf"(?<!\d){re.escape(form)}(?!\d)", text) for form in _number_forms(value)
        )
    return _normalize(str(value)) in _normalize(quote)


def _number_forms(value: float) -> set[str]:
    if float(value).is_integer():
        return {str(int(value))}
    plain = repr(float(value))
    return {plain, plain.replace(".", ",")}


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
