"""Shared tokenization helper for evaluation metrics and judges.

Extracts lowercase whitespace-delimited tokens as a set from text, used by both
the baseline answer-correctness metric (Jaccard similarity) and the judge fallback
(keyword overlap).
"""

from __future__ import annotations


def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase whitespace-delimited tokens."""
    return {tok for tok in text.lower().split() if tok}
