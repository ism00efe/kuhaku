"""Capability resolution: one identical loop for every decision point -- LLM backend,
embedding backend, device, retrieval mode, store.

Probe the environment, enumerate the candidates usable *right now* (no install, no
download), then branch on the count: zero -> state the gap and (if required) fail;
exactly one -> use it silently but announce it; more than one -> ask if a human is at the
terminal, otherwise pick the safe option and flag that a decision was skipped.

The loop carries no tool-specific knowledge. Each :class:`Adapter` reports only its own
candidates and their :class:`Cost`; the resolver never branches on a candidate id.
"""

from __future__ import annotations

from ._auto import AUTO, auto_enabled
from .cost import Candidate, Cost
from .environment import (
    FIELDS_CONSUMED,
    Environment,
    Fingerprint,
    fingerprint,
    probe_environment,
)
from .memory import JsonMemory, Memory
from .registry import Adapter, Registry
from .resolver import Resolution, activate, resolve
from .ui import UI, ConsoleUI

__all__ = [
    "AUTO",
    "auto_enabled",
    "Cost",
    "Candidate",
    "Adapter",
    "Registry",
    "Resolution",
    "resolve",
    "activate",
    "UI",
    "ConsoleUI",
    "Environment",
    "Fingerprint",
    "FIELDS_CONSUMED",
    "probe_environment",
    "fingerprint",
    "Memory",
    "JsonMemory",
]
