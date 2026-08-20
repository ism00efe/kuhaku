"""Optional LLM-as-judge faithfulness evaluation.

Disabled by default everywhere in this package -- ``EvaluationRunner`` only invokes an
evaluator here if one is explicitly passed to it (see ``runner.py``).
"""

from __future__ import annotations

import logging
import re

from kuhaku.core.llm.base import LLMProvider

from ._tokenization import tokenize

logger = logging.getLogger("kuhaku.evaluation.judges")

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict faithfulness judge for a RAG system. Given a question, retrieved "
    "context, and a generated answer, score how well the answer is supported by the "
    "context alone, on a scale from 0.0 (entirely unsupported / fabricated) to 1.0 "
    "(fully supported by the context). Respond with ONLY the numeric score."
)

_SCORE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")


def _keyword_overlap_fallback(context_chunks: list[str], generated_answer: str) -> float:
    """Deterministic fallback: fraction of the answer's tokens also present in context."""

    answer_tokens = tokenize(generated_answer)
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for chunk in context_chunks:
        context_tokens |= tokenize(chunk)
    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


class LLMFaithfulnessEvaluator:
    """Scores whether a generated answer is faithful to its retrieved context.

    Falls back to a deterministic keyword-overlap score whenever the LLM call fails or
    returns something unparsable, so a flaky provider never crashes an evaluation run.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def evaluate(self, question: str, context_chunks: list[str], generated_answer: str) -> float:
        context_block = "\n---\n".join(context_chunks)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Answer: {generated_answer}\n\n"
            "Faithfulness score (0.0-1.0):"
        )
        try:
            response = self._llm_provider.generate(_JUDGE_SYSTEM_PROMPT, user_prompt)
            match = _SCORE_PATTERN.search(response)
            if match is None:
                raise ValueError(f"no numeric score found in judge response: {response!r}")
            score = float(match.group(1))
        except Exception as exc:
            logger.warning("LLM faithfulness judge failed, using keyword-overlap fallback: %s", exc)
            return _keyword_overlap_fallback(context_chunks, generated_answer)
        return max(0.0, min(1.0, score))
