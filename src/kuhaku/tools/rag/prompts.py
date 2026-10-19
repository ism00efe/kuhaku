"""Prompt templates.

The system prompt is layered: a fixed, framework-owned safety/grounding core (instruction
precedence, [DOC] data-marking, the canary rule, grounding+abstention, mandatory
citations, contradiction handling, masked-value preservation) that every rendering
includes unconditionally, plus a small set of caller-owned layers -- persona, output-
language policy, answer-format preference, and a worked citation example -- that default
to neutral, English text but can be overridden independently via ``load_system_prompt``.
Citations use the ``[S1]..[Sk]`` tags injected into the context. Grounding rules are
explicit to reduce hallucination and to force citations.

Prompt-version tracking for eval A/B comparison and for production reproducibility
(cache/audit/response metadata) both moved to ``RAGSettings.eval_prompt_version`` /
``RAGSettings.prod_prompt_version`` -- this module no longer defines its own
``PROMPT_VERSION`` constant.

Category 2: the system prompt text itself lives in ``prompts/system_prompt.txt``
(this package's own data directory) rather than inline here, so it can be edited/versioned
without a code change. Loaded once at import time via a path resolved relative to this
module's file (``Path(__file__).parent``), not the process's CWD, so `kuhaku` works
the same whether run from source in this repo or `pip install`ed standalone into a
different project. A missing file fails loudly at startup instead of silently serving a
broken prompt.

Guard v2 hardening: retrieved chunks are wrapped in ``[DOC]...[/DOC]`` markers, and the
system prompt declares that marked text is data, never instructions, and that a canary
token must never appear in the model's output. This hardening ships unconditionally
(not gated by ``Settings.guard_enabled``) -- it costs nothing when the output-side canary
check is disabled, and avoids coupling this module to ``config.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

from kuhaku.core.prompts import (
    PromptRef,
    PromptRegistry,
    PromptTemplate,
    RenderedPrompt,
)
from kuhaku.core.security import CANARY_TOKEN
from kuhaku.tools.rag.models import RetrievedChunk

from .messages import (
    DEFAULT_ENGINE_MESSAGES,
    QUESTION_LABEL,
    SOURCES_LABEL,
    EngineMessages,
)

# Exported so nothing downstream (the UI's confidence badge, tests) carries a copied
# literal that could silently drift from what the model is actually instructed to say —
# same reasoning as `security.REFUSAL_MESSAGE`. Injectable via
# `load_system_prompt`'s `abstention_phrase` parameter for callers that need a different
# language/wording than this default.
ABSTENTION_PHRASE = "I don't have enough information to answer this question."

# -- Caller-owned layer defaults ----------------------------------------------------
# Neutral, English, domain-free: no industry, no product, no company, no forced output
# language. These are the values `load_system_prompt()` renders with when a caller
# overrides nothing -- see the framework/caller split documented on `load_system_prompt`.

DEFAULT_PERSONA = (
    "You are a general-purpose assistant that answers questions using only the "
    "reference material supplied with each question."
)

DEFAULT_LANGUAGE_POLICY = (
    "## Output language\n"
    "Answer in the same language the user's question was asked in. You may reason "
    "internally in any language, but the visible answer must match the question's "
    "language."
)

DEFAULT_FORMAT_PREFERENCE = (
    "## Format\n"
    "Answer as directly and concisely as the question allows. Use a numbered or "
    "step-by-step list only when the sources themselves describe a sequence of steps; "
    "otherwise prefer clear prose."
)

# Built from the same QUESTION_LABEL/SOURCES_LABEL constants build_user_prompt() uses,
# so the example's section labels match the real prompt layout by default. This sync
# only covers the *defaults* -- a caller who overrides EngineMessages.question_label/
# sources_label without also overriding this `example` layer will get an example whose
# labels no longer match the live prompt (the two injection points are intentionally
# independent, not mechanically coupled).
DEFAULT_EXAMPLE = (
    "## Example\n"
    f"{QUESTION_LABEL}\n"
    "What is the return window for online orders, and are opened items eligible?\n\n"
    f"{SOURCES_LABEL}\n"
    "[S1] (type=policy, title=Returns Policy)\n"
    "[DOC]\n"
    "Online orders may be returned within 30 days of delivery for a full refund.\n"
    "[/DOC]\n\n"
    "[S2] (type=faq, title=General FAQ)\n"
    "[DOC]\n"
    "Opened items are eligible for return as long as all original packaging is "
    "included.\n"
    "[/DOC]\n\n"
    "Expected answer:\n"
    '"Online orders can be returned within 30 days of delivery [S1]. Opened items are '
    'still eligible for return as long as the original packaging is included [S2]."'
)

# Matches exactly the labels kuhaku.core.sanitization.sanitize_text() emits (EMAIL,
# TOKEN, IP, CARD, NATIONAL_ID, PHONE) -- verified against that module rather than
# assumed. The leading " such as " is baked into the value (not the template) so an
# empty override collapses the sentence cleanly instead of leaving "placeholders . These
# mean..." behind -- see the empty-layer handling in `load_system_prompt`.
DEFAULT_MASKED_PLACEHOLDERS = " such as [EMAIL], [TOKEN], [IP], [CARD], [NATIONAL_ID], [PHONE]"

# Public alias: AssistantService.reconfigure() writes an admin-edited prompt back
# to this exact path, so it must be importable rather than re-hardcoded elsewhere.
SYSTEM_PROMPT_PATH = _SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"
)

# Registry name and version of the built-in system prompt. The version is the
# template's own (bump it when system_prompt.txt changes); it is independent of
# RAGEngine's `system_prompt_version`, the deployment label that also keys the QA cache.
SYSTEM_PROMPT_NAME = "rag.system"
SYSTEM_PROMPT_TEMPLATE_VERSION = "1"

# Every marker the template must contain. An override registered under
# SYSTEM_PROMPT_NAME must declare the same set: rendering passes all seven, and a
# template without {{CANARY_TOKEN}} would silently drop the canary rule.
SYSTEM_PROMPT_VARIABLES = frozenset(
    {
        "ABSTENTION_PHRASE",
        "CANARY_TOKEN",
        "PERSONA",
        "LANGUAGE_POLICY",
        "FORMAT_PREFERENCE",
        "EXAMPLE",
        "MASKED_PLACEHOLDERS",
    }
)


def builtin_prompts() -> PromptRegistry:
    """kuhaku's built-in RAG prompts, read fresh from this package's prompt files.

    Read on every call rather than cached at import, matching what
    ``load_system_prompt`` has always done, so a hot-edited ``system_prompt.txt`` is
    picked up by the next render.
    """

    raw = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    # Checked against the *raw* template (before any substitution) so a future rename or
    # typo of a {{...}} marker in system_prompt.txt fails loudly here instead of quietly
    # shipping that layer's text nowhere and a literal "{{FOO}}" to the model.
    missing = sorted(name for name in SYSTEM_PROMPT_VARIABLES if "{{" + name + "}}" not in raw)
    if missing:
        names = ", ".join("{{" + name + "}}" for name in missing)
        raise ValueError(f"system_prompt.txt is missing expected placeholder(s): {names}")
    return PromptRegistry(
        [
            PromptTemplate(
                SYSTEM_PROMPT_NAME,
                SYSTEM_PROMPT_TEMPLATE_VERSION,
                raw,
                variables=SYSTEM_PROMPT_VARIABLES,
                description="Layered RAG system prompt: caller-owned layers over the "
                "framework's safety/grounding core.",
            )
        ]
    )


def render_system_prompt(
    *,
    prompts: PromptRegistry | None = None,
    abstention_phrase: str = ABSTENTION_PHRASE,
    persona: str = DEFAULT_PERSONA,
    language_policy: str = DEFAULT_LANGUAGE_POLICY,
    format_preference: str = DEFAULT_FORMAT_PREFERENCE,
    example: str = DEFAULT_EXAMPLE,
    masked_placeholders: str = DEFAULT_MASKED_PLACEHOLDERS,
) -> RenderedPrompt:
    """:func:`load_system_prompt`, plus the :class:`~kuhaku.core.prompts.PromptRef` of
    the template it was rendered from.

    ``prompts`` is a caller's registry: a template registered there under
    :data:`SYSTEM_PROMPT_NAME` (its latest version, or the pinned one) replaces the
    built-in ``system_prompt.txt``. Such a template owns the safety core the same way
    ``RAG(system_prompt=...)`` does, except that it must still contain all seven
    markers -- including ``{{CANARY_TOKEN}}`` -- or rendering fails.
    """

    template = PromptRegistry.chain(prompts, builtin_prompts()).get(SYSTEM_PROMPT_NAME)
    rendered = template.render(
        ABSTENTION_PHRASE=abstention_phrase,
        CANARY_TOKEN=CANARY_TOKEN,
        PERSONA=persona,
        LANGUAGE_POLICY=language_policy,
        FORMAT_PREFERENCE=format_preference,
        EXAMPLE=example,
        MASKED_PLACEHOLDERS=masked_placeholders,
    )
    # An empty caller-supplied layer (e.g. persona="") must not leave a stray run of
    # blank lines behind where its heading+body used to be.
    text = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    return RenderedPrompt(text, template.ref)


def load_system_prompt(
    *,
    abstention_phrase: str = ABSTENTION_PHRASE,
    persona: str = DEFAULT_PERSONA,
    language_policy: str = DEFAULT_LANGUAGE_POLICY,
    format_preference: str = DEFAULT_FORMAT_PREFERENCE,
    example: str = DEFAULT_EXAMPLE,
    masked_placeholders: str = DEFAULT_MASKED_PLACEHOLDERS,
    prompts: PromptRegistry | None = None,
) -> str:
    """Render the system prompt template with each layer substituted.

    ``system_prompt.txt`` is split into a fixed, framework-owned safety/grounding core
    (instruction precedence, [DOC] data-marking, the canary rule, grounding+abstention,
    mandatory citations, contradiction handling, masked-value preservation) and a set of
    caller-owned layers this function parameterizes: ``persona``, ``language_policy``,
    ``format_preference``, ``example``, and the ``masked_placeholders`` list named in the
    (still fixed) masked-value rule. Every parameter defaults to a neutral, English,
    domain-free constant, so ``load_system_prompt()`` with no arguments renders a
    complete prompt -- the safety core is not itself a parameter and cannot be dropped
    through this signature. ``prompts`` swaps the template itself -- see
    :func:`render_system_prompt`.

    All seven markers ({{ABSTENTION_PHRASE}}, {{CANARY_TOKEN}}, and the five caller
    layers) are substituted in a single left-to-right pass over the raw template text
    (:meth:`~kuhaku.core.prompts.PromptTemplate.render`). That single pass is what makes
    the substitution safe against caller-supplied content that itself contains
    ``{{...}}`` (e.g. a crafted persona string): inserted text is never rescanned, so
    such content is written out as inert literal characters and can never trigger a
    second substitution that would, for example, splice the real ``CANARY_TOKEN`` into
    the rendered prompt.
    """

    return render_system_prompt(
        prompts=prompts,
        abstention_phrase=abstention_phrase,
        persona=persona,
        language_policy=language_policy,
        format_preference=format_preference,
        example=example,
        masked_placeholders=masked_placeholders,
    ).text


_DEFAULT_SYSTEM_PROMPT = render_system_prompt()
SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT.text
SYSTEM_PROMPT_REF: PromptRef = _DEFAULT_SYSTEM_PROMPT.ref


# Markup the prompt itself uses to delimit sources. Retrieved text that contains it could
# close its own [DOC] block early and open a forged one ("...[/DOC] [S9] (type=policy)
# [DOC] new instructions"), escaping the data marking. Matched loosely -- case and inner
# whitespace vary -- and rewritten with parentheses, which the prompt never uses for
# markup, so the text stays readable but can no longer act as a boundary.
_DOC_MARKER_RE = re.compile(r"\[\s*(/?)\s*doc\s*\]", re.IGNORECASE)
_SOURCE_TAG_RE = re.compile(r"\[\s*s\s*(\d+)\s*\]", re.IGNORECASE)


def _neutralize_markup(text: str) -> str:
    text = _DOC_MARKER_RE.sub(lambda m: f"({m.group(1)}DOC)", text)
    return _SOURCE_TAG_RE.sub(lambda m: f"(S{m.group(1)})", text)


def _format_sources(retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for i, item in enumerate(retrieved, start=1):
        c = item.chunk
        doc_type = _neutralize_markup(c.doc_type)
        title = _neutralize_markup(c.title)
        text = _neutralize_markup(c.text.strip())
        blocks.append(f"[S{i}] (type={doc_type}, title={title})\n[DOC]\n{text}\n[/DOC]")
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
    # engine's out-of-range guard drops it), but citation verification would flag it
    # as an unverified citation the model never actually used.
    example_tags = ", ".join(f"[S{i}]" for i in range(1, len(retrieved) + 1)) or "[S1]"
    return (
        f"{messages.question_label}\n{question}\n\n"
        f"{messages.sources_label}\n{sources}\n\n"
        + messages.cite_instruction_template.format(example_tags=example_tags)
    )
