"""Prompt templates.

The knowledge base is English; the assistant answers in Turkish and cites sources by the
``[S1]..[Sk]`` tags injected into the context. Grounding rules are explicit to reduce
hallucination and to force citations.

Prompt-version tracking for eval A/B comparison and for production reproducibility
(cache/audit/response metadata) both moved to ``RAGSettings.eval_prompt_version`` /
``RAGSettings.prod_prompt_version`` (D42) -- this module no longer defines its own
``PROMPT_VERSION`` constant.

FR1 (Category 2): the system prompt text itself lives in ``prompts/system_prompt.txt``
(this package's own data directory) rather than inline here, so it can be edited/versioned
without a code change. Loaded once at import time via a path resolved relative to this
module's file (``Path(__file__).parent``), not the process's CWD, so `kuhaku` works
the same whether run from source in this repo or `pip install`ed standalone into a
different project. A missing file fails loudly at startup instead of silently serving a
broken prompt.

FR3 (guard v2, D39): retrieved chunks are wrapped in ``[DOC]...[/DOC]`` markers, and the
system prompt declares that marked text is data, never instructions, and that a canary
token must never appear in the model's output. This hardening ships unconditionally
(not gated by ``Settings.guard_enabled``) -- it costs nothing when the output-side canary
check is disabled, and avoids coupling this module to ``config.py``.
"""

from __future__ import annotations

from pathlib import Path

from kuhaku.tools.rag.models import RetrievedChunk
from kuhaku.core.security import CANARY_TOKEN

from .messages import DEFAULT_ENGINE_MESSAGES, EngineMessages

# Exported so nothing downstream (the UI's confidence badge, tests) carries a copied
# literal that could silently drift from what the model is actually instructed to say —
# same reasoning as `security.REFUSAL_MESSAGE` (see DECISIONS.md D22). Injectable via
# `load_system_prompt`'s `abstention_phrase` parameter for callers that need a different
# language/wording than this default.
ABSTENTION_PHRASE = "I don't have enough information to answer this question."

# Public alias (D46): AssistantService.reconfigure() writes an admin-edited prompt back
# to this exact path, so it must be importable rather than re-hardcoded elsewhere.
SYSTEM_PROMPT_PATH = _SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"
)


def load_system_prompt(abstention_phrase: str = ABSTENTION_PHRASE) -> str:
    """Render the system prompt template with the given abstention phrase substituted."""

    text = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    text = text.replace("{{ABSTENTION_PHRASE}}", abstention_phrase)
    text = text.replace("{{CANARY_TOKEN}}", CANARY_TOKEN)
    return text.strip()


SYSTEM_PROMPT = load_system_prompt()


def _format_sources(retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for i, item in enumerate(retrieved, start=1):
        c = item.chunk
        blocks.append(
            f"[S{i}] (type={c.doc_type}, title={c.title})\n[DOC]\n{c.text.strip()}\n[/DOC]"
        )
    return "\n\n".join(blocks)


def build_user_prompt(
    question: str,
    retrieved: list[RetrievedChunk],
    messages: EngineMessages = DEFAULT_ENGINE_MESSAGES,
) -> str:
    """Assemble the grounded user prompt from the question and retrieved context."""

    sources = _format_sources(retrieved) if retrieved else messages.no_sources_fallback
    # The citation-format reminder must only name tags that actually exist in this
    # prompt's sources block. A hardcoded "[S1], [S2]" would put a dangling [S2] into
    # the prompt itself whenever only one chunk was retrieved -- harmless today (the
    # engine's out-of-range guard drops it), but FR4 citation verification would flag it
    # as an unverified citation the model never actually used.
    example_tags = ", ".join(f"[S{i}]" for i in range(1, len(retrieved) + 1)) or "[S1]"
    return (
        f"{messages.question_label}\n{question}\n\n"
        f"{messages.sources_label}\n{sources}\n\n"
        + messages.cite_instruction_template.format(example_tags=example_tags)
    )
