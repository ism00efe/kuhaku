"""Recording which prompts produced an LLM call.

Keyed by the role the prompt played in the call (``system``, ``user``), because one call
can be built from several templates and a backend query should not have to know which
template names a tool happens to use. The same role -> ref shape is used on spans
(flattened into attributes), in audit records (``metadata["prompts"]``) and in tool
results, so the three can be joined.
"""

from __future__ import annotations

from ..observability.tracing import set_current_span_attributes
from .template import PromptRef

PROMPT_ATTRIBUTE_PREFIX = "kuhaku.prompt"


def prompt_span_attributes(**refs: PromptRef | None) -> dict[str, str]:
    """``{"kuhaku.prompt.<role>.name"|".version"|".hash": ...}`` for each non-``None`` ref."""

    attributes: dict[str, str] = {}
    for role, ref in refs.items():
        if ref is None:
            continue
        prefix = f"{PROMPT_ATTRIBUTE_PREFIX}.{role}"
        attributes[f"{prefix}.name"] = ref.name
        attributes[f"{prefix}.version"] = ref.version
        attributes[f"{prefix}.hash"] = ref.content_hash
    return attributes


def prompt_metadata(**refs: PromptRef | None) -> dict[str, dict[str, str]]:
    """``{role: {"name", "version", "hash"}}`` -- for audit metadata and tool results."""

    return {role: ref.as_dict() for role, ref in refs.items() if ref is not None}


def record_prompts(**refs: PromptRef | None) -> None:
    """Annotate the current span with :func:`prompt_span_attributes`. Best-effort and a
    no-op without a recording span, like every other span annotation in kuhaku."""

    set_current_span_attributes(**prompt_span_attributes(**refs))
