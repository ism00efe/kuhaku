"""The two data shapes every adapter speaks in: :class:`Cost` and :class:`Candidate`.

An adapter reports only its own candidates and each candidate's cost. The resolver never
inspects a candidate id -- it branches on how many candidates are ``ready`` and on
``safety_rank``, nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Cost:
    """What choosing a candidate costs. Every field defaults to the cheap answer so an
    adapter only states what is actually true of its candidate."""

    install_required: bool = False
    """A package must be installed before this candidate can be used."""
    download_required: bool = False
    """Model weights must be fetched before this candidate can be used."""
    download_bytes: int | None = None
    """Approximate download size; ``None`` means unknown."""
    network_per_use: bool = False
    """Every use makes a network call."""
    sends_document_text: bool = False
    """The full text of every ingested document leaves the machine (ingest-time)."""
    monetary: bool = False
    """Billed per use."""
    note: str = ""
    """One short human-readable clause -- shown in questions and announcements, and, when
    it begins ``pip install``, used as the install command."""


@dataclass(frozen=True)
class Candidate:
    """One concrete way to satisfy a decision."""

    id: str
    """Stable, machine-readable, lowercase -- the backend's short name."""
    kind: str
    """``"llm"`` | ``"embedding"`` | ``"store"`` | ``"device"`` | ``"retrieval"`` | future."""
    label: str
    """Short human-facing text."""
    cost: Cost
    ready: bool
    """Usable right now, with no install and no download."""
    safety_rank: int
    """Lower = safer; the ordering used for non-interactive selection."""
    activate: Callable[[], Any]
    """Constructs the backend. Never called during probing or resolution -- only by
    :func:`kuhaku.core.resolve.resolver.activate`, after selection and consent."""
