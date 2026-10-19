"""Output-side validation for the v2 guard pipeline.

Runs on a tool's generated answer, after citation extraction but before the response
reaches the caller: citation grounding (annotate only), canary/extraction detection
(block), PII egress (block). Composed and invoked from ``tools/rag/engine.py``, only when
a `GuardPipeline` is configured (`guard_enabled=True`) -- but the checks themselves are
tool-agnostic: they operate on plain context strings and citation tags, not on any
RAG-specific type, so a different tool built on kuhaku can reuse them unchanged.

Never raises: any internal failure here returns the original answer text unmodified,
mirroring `RAGEngine._flag_unverified_citations`' exact fail-open idiom — a bug in
this module must never break an otherwise-good answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..observability.metrics import record_guard_degradation
from ..sanitization import (  # _iban_ok re-exported: it lived here before
    _CARD_RE,
    _IBAN_RE,
    _TCKN_RE,
    _iban_ok,  # noqa: F401
    _luhn_ok,
    _tckn_ok,
    _valid_iban_prefix,
)

logger = logging.getLogger(__name__)

SAFE_FALLBACK = "A technical issue occurred, please try again."

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
# A segment left over after splitting that is nothing but citation tags (e.g. the LLM
# wrote "claim. [S1]" -- the naive split above would isolate "[S1]" as its own
# content-free segment). Merged back into the preceding segment so the tag stays
# attached to the sentence it actually supports.
_TAG_ONLY_RE = re.compile(r"^(\[S\d+\]\s*)+$")

_CANARY_MIN_SUBSTRING = 8

def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_into_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping a trailing citation-tag-only segment
    (e.g. "[S1]") attached to the sentence before it rather than isolated on its own."""

    raw_segments = _SENTENCE_SPLIT_RE.split(text)
    merged: list[str] = []
    for segment in raw_segments:
        if merged and _TAG_ONLY_RE.match(segment.strip()):
            merged[-1] = f"{merged[-1]} {segment}"
        else:
            merged.append(segment)
    return merged


def check_citation_grounding(
    text: str,
    citation_tags: list[str],
    contexts: list[str],
    *,
    threshold: float,
) -> list[str]:
    """Return the tags (e.g. ``"S2"``) of citations whose citing sentence has token-set
    Jaccard similarity below ``threshold`` against its referenced context string.

    ``citation_tags`` are the ``[S#]``-style tags actually cited (e.g. ``["S1", "S2"]``);
    ``contexts`` is the ordered list of context strings a tag's ``N`` indexes into
    (``contexts[N - 1]``) -- e.g. retrieved chunk text, one entry per source. Only tags
    already mapped to a valid, in-range source are considered — an out-of-range ``[S#]``
    is the calling tool's own citation-validity concern (e.g.
    `RAGEngine._flag_unverified_citations`), not this one's.
    """

    sentences = _split_into_sentences(text)
    ungrounded: list[str] = []
    for tag in citation_tags:
        try:
            idx = int(tag[1:])
        except ValueError:
            continue
        if not (1 <= idx <= len(contexts)):
            continue

        marker = f"[{tag}]"
        matching_sentences = [s for s in sentences if marker in s]
        if not matching_sentences:
            continue

        context_tokens = _tokenize(contexts[idx - 1])
        best = max(_jaccard(_tokenize(s), context_tokens) for s in matching_sentences)
        if best < threshold:
            ungrounded.append(tag)
    return ungrounded


def annotate_ungrounded_citations(text: str, ungrounded_tags: list[str]) -> str:
    """Insert ``[unverified]`` immediately after each flagged ``[S#]`` occurrence."""

    for tag in ungrounded_tags:
        marker = f"[{tag}]"
        text = text.replace(marker, f"{marker}[unverified]")
    return text


def detect_canary(text: str, canary: str) -> bool:
    """True if the full canary token, or any >=8-char substring of it, appears in text."""

    if not canary or not text:
        return False
    if canary in text:
        return True
    for start in range(len(canary) - _CANARY_MIN_SUBSTRING + 1):
        if canary[start : start + _CANARY_MIN_SUBSTRING] in text:
            return True
    return False


def check_pii_egress(text: str, contexts: list[str]) -> list[str]:
    """Return the PII categories (``"PAN"``/``"NATIONAL_ID"``/``"IBAN"``) found in ``text``
    whose matched value does not appear verbatim in ``contexts`` (e.g. retrieved chunk
    text).

    `sanitization.py` masks cards, national ids and IBANs (spaced or not) at ingestion,
    so a legitimately-retrieved chunk should not contain a raw one and the "not in
    contexts" branch is practically always true: any valid-checksum value the LLM emits
    is flagged. The check stays context-aware anyway, as a defence against a future
    sanitization gap for some format. The IBAN pattern and mod-97 validator are the
    sanitizer's own, so the two cannot drift apart.
    """

    corpus = "\n".join(contexts)
    categories: list[str] = []

    for match in _CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if _luhn_ok(digits) and match.group(0) not in corpus:
            categories.append("PAN")

    for match in _TCKN_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if _tckn_ok(digits) and match.group(0) not in corpus:
            categories.append("NATIONAL_ID")

    for match in _IBAN_RE.finditer(text):
        candidate = _valid_iban_prefix(match.group(0))
        if candidate is not None and candidate not in corpus:
            categories.append("IBAN")

    seen: set[str] = set()
    deduped: list[str] = []
    for category in categories:
        if category not in seen:
            seen.add(category)
            deduped.append(category)
    return deduped


@dataclass(frozen=True)
class OutputGuardResult:
    text: str
    ungrounded_citations: list[str]
    canary_detected: bool
    pii_egress_detected: bool
    pii_egress_categories: list[str]  # never raw values, only category labels
    blocked: bool  # True => text was replaced with the safe fallback message


def evaluate_output(
    text: str,
    citation_tags: list[str],
    contexts: list[str],
    *,
    canary: str,
    citation_grounding_threshold: float,
) -> OutputGuardResult:
    """Compose the three output checks: grounding (annotate only) -> canary (block) ->
    PII egress (block, only checked if canary did not already fire).

    ``citation_tags``/``contexts`` are the same generic shapes as
    :func:`check_citation_grounding` -- see there for how a tag indexes into ``contexts``.
    """

    try:
        ungrounded = check_citation_grounding(
            text, citation_tags, contexts, threshold=citation_grounding_threshold
        )
        annotated = annotate_ungrounded_citations(text, ungrounded) if ungrounded else text

        if detect_canary(annotated, canary):
            logger.critical(
                "guard v2: canary token detected in LLM output -- prompt extraction "
                "suspected",
                extra={"canary_detected": True},
            )
            return OutputGuardResult(
                text=SAFE_FALLBACK,
                ungrounded_citations=ungrounded,
                canary_detected=True,
                pii_egress_detected=False,
                pii_egress_categories=[],
                blocked=True,
            )

        pii_categories = check_pii_egress(annotated, contexts)
        if pii_categories:
            logger.critical(
                "guard v2: PII egress detected in LLM output (categories=%s), not "
                "present in any retrieved chunk",
                pii_categories,
                extra={"pii_egress_detected": True},
            )
            return OutputGuardResult(
                text=SAFE_FALLBACK,
                ungrounded_citations=ungrounded,
                canary_detected=False,
                pii_egress_detected=True,
                pii_egress_categories=pii_categories,
                blocked=True,
            )

        return OutputGuardResult(
            text=annotated,
            ungrounded_citations=ungrounded,
            canary_detected=False,
            pii_egress_detected=False,
            pii_egress_categories=[],
            blocked=False,
        )
    except Exception:
        logger.exception("output guard failed; returning the original answer unmodified")
        record_guard_degradation("output_guard")
        return OutputGuardResult(
            text=text,
            ungrounded_citations=[],
            canary_detected=False,
            pii_egress_detected=False,
            pii_egress_categories=[],
            blocked=False,
        )
