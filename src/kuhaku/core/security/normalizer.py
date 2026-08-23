"""Input normalization — the first layer of the v2 guard pipeline.

Unicode NFKC normalization, zero-width character removal, homoglyph folding, known
injection-delimiter stripping, and repeated-punctuation collapse. The normalized text is
the ONLY text passed downstream to the classifier (`classifier.py`) and the datamarked
prompt template — raw text is never scored or embedded, only hashed for the audit log
(`audit.py`).

No dependency on `config.py` or `observability` on purpose: this module is a pure
text -> text function, matching `rag/prompts.py`'s own zero-dependency-on-config
precedent. Callers (`guard.py`) are responsible for logging DEBUG when drift exceeds the
configured tolerance.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Characters with no visible glyph, commonly used to split up or hide injection payloads
# from naive substring/regex matching.
_ZERO_WIDTH = (
    "​"  # ZERO WIDTH SPACE
    "‌"  # ZERO WIDTH NON-JOINER
    "‍"  # ZERO WIDTH JOINER
    "﻿"  # ZERO WIDTH NO-BREAK SPACE / BOM
    "⁠"  # WORD JOINER
    "⁡"  # FUNCTION APPLICATION
    "⁢"  # INVISIBLE TIMES
    "⁣"  # INVISIBLE SEPARATOR
    "⁤"  # INVISIBLE PLUS
)
_ZERO_WIDTH_TABLE = {ord(ch): None for ch in _ZERO_WIDTH}

# Visually-confusable characters (Cyrillic/Greek look-alikes of Latin letters) folded to
# their ASCII equivalent so a homoglyph-obfuscated word ("ignоre" with a Cyrillic 'о')
# matches the same keyword/regex signals as its plain-ASCII form. Not exhaustive — new
# confusables can be added here as they're observed; this is a documented, extensible
# table, not a claim of completeness (residual mixed-script text is instead caught by
# classifier.py's script-mixing feature).
_HOMOGLYPHS: dict[str, str] = {
    # Cyrillic -> Latin
    "а": "a", "А": "A",
    "е": "e", "Е": "E",
    "о": "o", "О": "O",
    "р": "p", "Р": "P",
    "с": "c", "С": "C",
    "х": "x", "Х": "X",
    "у": "y", "У": "Y",
    "і": "i", "І": "I",
    "ѕ": "s", "Ѕ": "S",
    "ј": "j", "Ј": "J",
    "к": "k", "К": "K",
    "м": "m", "М": "M",
    "н": "h", "Н": "H",
    "т": "t", "Т": "T",
    "в": "b", "В": "B",
    # Greek -> Latin
    "α": "a", "Α": "A",
    "β": "b", "Β": "B",
    "ο": "o", "Ο": "O",
    "ρ": "p", "Ρ": "P",
    "ν": "v", "Ν": "N",
    "κ": "k", "Κ": "K",
    "τ": "t", "Τ": "T",
    "ι": "i", "Ι": "I",
    "υ": "u", "Υ": "Y",
}
_HOMOGLYPH_TABLE = {ord(k): v for k, v in _HOMOGLYPHS.items()}

# Known injection-template delimiters — chat-template turn markers and Markdown-style
# section breaks attackers use to try to smuggle a fake "system" turn into the message.
# Stripped (not just flagged) so the classifier/LLM never sees them as structure.
_DELIMITERS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "</system>",
    "</user>",
    "</assistant>",
    "[SYSTEM]",
    "[USER]",
    "[ASSISTANT]",
    "---",
    "###",
)
# Longest-first so e.g. "[SYSTEM]" is removed whole rather than leaving stray brackets
# behind if a shorter pattern were checked first (none currently overlap, but this keeps
# the property true as the table grows).
_DELIMITERS_SORTED = sorted(_DELIMITERS, key=len, reverse=True)
_DELIMITER_RE = re.compile(
    "|".join(re.escape(d) for d in _DELIMITERS_SORTED), re.IGNORECASE
)

_REPEATED_PUNCT_RE = re.compile(r"([!?.,;:])\1{2,}")


@dataclass(frozen=True)
class NormalizationResult:
    """The normalized text plus enough metadata to compute drift for risk scoring."""

    text: str
    raw_length: int
    normalized_length: int
    drift: int


def normalize(raw: str) -> NormalizationResult:
    """Apply the full normalization pipeline to ``raw``.

    Order: NFKC -> strip zero-width chars -> fold homoglyphs -> strip known delimiters ->
    collapse repeated punctuation. Each step operates on the previous step's output.
    """

    if not raw:
        return NormalizationResult(text=raw, raw_length=0, normalized_length=0, drift=0)

    text = unicodedata.normalize("NFKC", raw)
    text = text.translate(_ZERO_WIDTH_TABLE)
    text = text.translate(_HOMOGLYPH_TABLE)
    text = _DELIMITER_RE.sub("", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)

    raw_length = len(raw)
    normalized_length = len(text)
    return NormalizationResult(
        text=text,
        raw_length=raw_length,
        normalized_length=normalized_length,
        drift=abs(raw_length - normalized_length),
    )
