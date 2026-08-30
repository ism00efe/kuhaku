"""Small formatting helpers shared by the resolver and the consent flow."""

from __future__ import annotations


def format_bytes(n: int | None) -> str:
    """Human size using decimal units (so 4_000_000_000 reads as ``4.0 GB``)."""

    if n is None:
        return "unknown size"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f} MB"
    return f"{n} bytes"
