"""VRAM -> model-size-class, the one hardware computation §9 says is safe to encode
(model family and version names are not, and are not kuhaku's business -- §10).
"""

from __future__ import annotations

from typing import Literal

# Fraction of VRAM reserved beyond the weights themselves, for context / KV cache.
# An estimate, not a measurement -- overridable via RAGSettings.vram_headroom.
VRAM_HEADROOM = 0.25

# Rough usable-VRAM budget per class, in GB. Estimates, not measurements.
_CLASS_BUDGET_GB: dict[str, float] = {
    "none": 0.0,
    "unknown": 0.0,
    "small": 5.0,
    "medium": 14.0,
    "large": 40.0,
}

SizeClass = Literal["tiny", "small", "medium", "large", "unknown"]


def recommended_size_class(vram_class: str, *, headroom: float = VRAM_HEADROOM) -> SizeClass:
    """A coarse model size class the machine can likely host. ``"unknown"`` when there is
    no GPU or its VRAM could not be read -- in which case guidance stays purely
    qualitative and no model name is ever produced (§10)."""

    if vram_class in ("none", "unknown") or vram_class not in _CLASS_BUDGET_GB:
        return "unknown"
    budget = _CLASS_BUDGET_GB[vram_class] * (1.0 - headroom)
    if budget < 3.0:
        return "tiny"
    if budget < 8.0:
        return "small"
    if budget < 24.0:
        return "medium"
    return "large"
