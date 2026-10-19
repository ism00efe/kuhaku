"""Optional LLM-as-judge faithfulness evaluation.

Disabled by default everywhere in this package -- ``EvaluationRunner`` only invokes an
evaluator here if one is explicitly passed to it (see ``runner.py``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from kuhaku.core.llm.base import LLMProvider

from ._tokenization import tokenize

logger = logging.getLogger("kuhaku.evaluation.judges")

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict faithfulness judge for a RAG system. Given a question, retrieved "
    "context, and a generated answer, score how well the answer is supported by the "
    "context alone, on a scale from 0.0 (entirely unsupported / fabricated) to 1.0 "
    "(fully supported by the context). Respond with ONLY a JSON object of the form "
    '{"score": <number>}.'
)

_SCORE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")
_JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*\}")


@dataclass(frozen=True)
class JudgeScore:
    """A faithfulness score plus where it came from. ``fallback`` is ``True`` when the
    judge LLM failed or answered unparsably and the keyword-overlap heuristic produced
    ``score`` instead -- a different measurement that must not be averaged together
    with real judge scores."""

    score: float
    fallback: bool = False


def parse_judge_score(response: str) -> float:
    """Extract the score from a judge response.

    A JSON object with a ``score`` key wins. Otherwise the *last* number is taken:
    judges that ignore the format tend to restate the scale first ("On a 0 to 1 scale,
    ... 0.9"), so the first number is the scale's lower bound, not the verdict.
    Raises ``ValueError`` when there is no number at all.
    """

    for candidate in reversed(_JSON_OBJECT_PATTERN.findall(response)):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("score"), (int, float)):
            return float(parsed["score"])
    numbers = _SCORE_PATTERN.findall(response)
    if not numbers:
        raise ValueError(f"no numeric score found in judge response: {response!r}")
    return float(numbers[-1])


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
        """The score alone -- see :meth:`evaluate_detailed` to know whether it is the
        judge's or the keyword-overlap fallback's."""

        return self.evaluate_detailed(question, context_chunks, generated_answer).score

    def evaluate_detailed(
        self, question: str, context_chunks: list[str], generated_answer: str
    ) -> JudgeScore:
        context_block = "\n---\n".join(context_chunks)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Answer: {generated_answer}\n\n"
            'Faithfulness score as {"score": <0.0-1.0>}:'
        )
        try:
            response = self._llm_provider.generate(_JUDGE_SYSTEM_PROMPT, user_prompt)
            score = parse_judge_score(response)
        except Exception as exc:
            logger.warning("LLM faithfulness judge failed, using keyword-overlap fallback: %s", exc)
            return JudgeScore(
                _keyword_overlap_fallback(context_chunks, generated_answer), fallback=True
            )
        return JudgeScore(max(0.0, min(1.0, score)))
