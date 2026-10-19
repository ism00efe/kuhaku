"""Prompt templates, a versioned registry, and per-call prompt tracking.

Tool-agnostic: a tool registers its built-in prompts in its own
:class:`PromptRegistry`, renders them through it, and records the :class:`PromptRef`
(name, version, content hash) of every prompt that went into an LLM call. A user
overrides a prompt by registering another version under the same name and handing that registry to the
tool; kuhaku's own code is never edited.

    from kuhaku.core.prompts import PromptRegistry, PromptTemplate

    registry = PromptRegistry([PromptTemplate("greet", "1", "Hello, {{name}}.")])
    registry.get("greet").render(name="Ada")   # 'Hello, Ada.'
"""

from .fence import fence, new_boundary, wrap
from .registry import PromptRegistry, version_key
from .template import (
    CUSTOM_VERSION,
    PromptConflictError,
    PromptError,
    PromptNotFoundError,
    PromptRef,
    PromptRenderError,
    PromptTemplate,
    PromptTemplateError,
    RenderedPrompt,
    content_hash,
    placeholders,
)
from .tracking import (
    PROMPT_ATTRIBUTE_PREFIX,
    prompt_metadata,
    prompt_span_attributes,
    record_prompts,
)

__all__ = [
    "CUSTOM_VERSION",
    "PROMPT_ATTRIBUTE_PREFIX",
    "PromptConflictError",
    "PromptError",
    "PromptNotFoundError",
    "PromptRef",
    "PromptRegistry",
    "PromptRenderError",
    "PromptTemplate",
    "PromptTemplateError",
    "RenderedPrompt",
    "content_hash",
    "fence",
    "new_boundary",
    "placeholders",
    "prompt_metadata",
    "prompt_span_attributes",
    "record_prompts",
    "version_key",
    "wrap",
]
