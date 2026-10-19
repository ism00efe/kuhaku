"""Fencing untrusted text inside a prompt.

Text a model must treat as data (a document, a user's question, a tool's output, query
rows) goes between two copies of a boundary marker that is random per call, so the text
cannot contain the closing marker and "end" the data early to pose as instructions. The
prompt then names the boundary: "Source text, between the two SOURCE-1a2b3c... markers".

    boundary, fenced = fence(text, "SOURCE")

A tool that reuses one boundary for several pieces of data (the agent fences every tool
result of a run with the same marker) uses :func:`new_boundary` once and
:func:`wrap` per piece; :func:`wrap` removes any copy of the boundary from the text,
so a piece that echoes something it was shown still cannot close the fence.
"""

from __future__ import annotations

import re
import secrets

__all__ = ["fence", "new_boundary", "wrap"]

_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9-]*$")


def new_boundary(label: str) -> str:
    """``"<LABEL>-<12 hex chars>"`` -- random, so data cannot guess it."""

    if not _LABEL_RE.match(label):
        raise ValueError(f"A fence label is upper-case letters, digits and '-'; got {label!r}.")
    return f"{label}-{secrets.token_hex(6)}"


def wrap(text: str, boundary: str) -> str:
    """``text`` between two ``<<<boundary>>>`` lines, with any copy of the boundary
    inside it removed."""

    return f"<<<{boundary}>>>\n{text.replace(boundary, '')}\n<<<{boundary}>>>"


def fence(text: str, label: str) -> tuple[str, str]:
    """``(boundary, fenced text)`` with a fresh boundary for ``label``."""

    boundary = new_boundary(label)
    return boundary, wrap(text, boundary)
