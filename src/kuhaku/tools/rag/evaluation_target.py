"""RAG-flavored ``TargetAdapter``: adds an ``llm_provider`` answer-generation fallback.

The generic ``kuhaku.evaluation.target.TargetAdapter`` has no notion of a prompt
template -- a bare retriever target with no ``answer_generator`` simply produces no
answer. ``RAGTargetAdapter`` restores this package's original convenience (give it an
``llm_provider`` and it will generate an answer using the RAG system/user prompts) by
subclassing the generic adapter, keeping the RAG prompt import out of the tool-agnostic
evaluation layer entirely.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from kuhaku.core.llm.base import LLMProvider
from kuhaku.evaluation.target import TargetAdapter

from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger("kuhaku.tools.rag.evaluation_target")


class RAGTargetAdapter(TargetAdapter):
    """``TargetAdapter`` with a RAG prompt-driven ``llm_provider`` fallback.

    Only reached for a bare retriever target (``retrieve()``/``search()``) with no
    ``answer_generator`` -- an ``ask()``-style target's own answer always wins, matching
    the generic adapter's precedence.
    """

    def __init__(
        self,
        target: Any,
        *,
        top_k: int = 5,
        answer_generator: Callable[[str, list[Any]], str | None] | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        super().__init__(target, top_k=top_k, answer_generator=answer_generator)
        self._llm_provider = llm_provider

    def _generate_answer(
        self,
        query: str,
        retrieved: list[Any],
        *,
        default: str | None,
        allow_fallback: bool = True,
    ) -> str | None:
        answer = super()._generate_answer(query, retrieved, default=default, allow_fallback=allow_fallback)
        if answer is not None or not allow_fallback or self._answer_generator is not None:
            return answer
        if self._llm_provider is not None:
            try:
                return self._llm_provider.generate(
                    SYSTEM_PROMPT, build_user_prompt(query, retrieved)
                )
            except Exception as exc:
                logger.warning("llm_provider generation failed for query=%r: %s", query, exc)
                return None
        return answer
