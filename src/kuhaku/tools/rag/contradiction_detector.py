"""Real-time contradiction detection over retrieved chunks.

Distinct from Faithfulness (``evaluation/metrics/faithfulness.py``): Faithfulness checks
whether the *generated answer* is supported by *some* retrieved chunk -- it would still
score an answer as "faithful" if the LLM picked a stale chunk while a corrected one sat
right next to it in the same retrieval. This module instead inspects the *retrieved
chunk set itself*, before generation, for chunks that disagree with each other on the
same topic (e.g. an old vs. a new regulation) -- a distinct, query-time-only signal.

Two-stage design, mirroring the task's own algorithm: cheap embedding-based topic
clustering (cosine similarity over already-normalized vectors, no new provider call)
narrows an O(n^2) chunk-pair space down to same-topic candidates, then only those
candidates go to the judge LLM (temperature=0, the same instance
``AssistantService._judge_llm`` already uses for Faithfulness/query rewriting)
for an actual contradiction verdict. Judge-call/JSON-parsing failure handling mirrors
``evaluation/faithfulness.py``'s ``LLMFaithfulnessEvaluator`` exactly.

Graceful degradation is enforced at two levels: a single candidate pair whose judge call
or JSON parse fails is simply skipped (not counted as a contradiction, but does not
abort other candidate pairs); ``RAGEngine._answer()`` wraps the whole ``detect()`` call
in its own ``try/except`` so a systemic failure (e.g. the judge LLM is entirely
unreachable) never delays or blocks the response (Constraint 5).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import combinations

from kuhaku.core.llm.base import LLMProvider
from kuhaku.tools.rag.models import RetrievedChunk

from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial fact-checking judge. You only output valid JSON, "
    "exactly as instructed, with no surrounding text."
)

# Generic default -- injectable via `prompt_template` for callers that need
# domain-specific wording (e.g. a particular document corpus or answer language).
DEFAULT_PROMPT_TEMPLATE = """You are checking two text excerpts for contradictions.

Text A: {text_a}

Text B: {text_b}

Do these two texts contain contradictory or conflicting information about the same topic?

Rules:
- A contradiction means they state different facts, numbers, dates, or requirements
  about the same subject.
- If they describe different scopes (e.g., one is about individuals, the other about
  organizations), that is NOT a contradiction.
- If one text provides a general rule and the other provides an exception, that is NOT
  a contradiction — it's a nuance.
- If you are unsure, answer YES (prefer false positives over false negatives).

Respond ONLY with a valid JSON object:
{{
  "contradiction": true/false,
  "topic": "brief topic description",
  "explanation": "brief explanation of what contradicts"
}}"""

# Formatted with a single ``pair_refs`` keyword (the joined "[S#]" tags).
DEFAULT_WARNING_TEMPLATE = (
    "⚠️ Sources with conflicting information on this topic exist. See {pair_refs}."
)


@dataclass(frozen=True)
class ContradictionPair:
    """One confirmed contradiction between two retrieved chunks."""

    chunk_a_id: str
    chunk_b_id: str
    topic: str
    explanation: str
    confidence: float


@dataclass(frozen=True)
class ContradictionResult:
    """Outcome of one :meth:`ContradictionDetector.detect` call."""

    has_contradiction: bool
    warning_message: str | None = None
    contradiction_pairs: list[ContradictionPair] = field(default_factory=list)


class ContradictionDetector:
    """Embedding-clustering + LLM-as-judge contradiction check over retrieved chunks."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        judge_llm: LLMProvider,
        *,
        similarity_threshold: float = 0.75,
        judge_system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        warning_template: str = DEFAULT_WARNING_TEMPLATE,
    ) -> None:
        self._embedder = embedder
        self._judge_llm = judge_llm
        self._similarity_threshold = similarity_threshold
        self._judge_system_prompt = judge_system_prompt
        self._prompt_template = prompt_template
        self._warning_template = warning_template

    def detect(self, chunks: list[RetrievedChunk]) -> ContradictionResult:
        """Check ``chunks`` (the current query's retrieved set) for contradictions."""

        if len(chunks) < 2:
            return ContradictionResult(has_contradiction=False)

        # [S#] indices mirror rag/prompts.py::_format_sources' own 1-indexed numbering
        # over the same retrieval-ordered list, so a reference built here always matches
        # the citation tags the LLM's response will use later.
        indexed = list(enumerate(chunks, start=1))

        vectors = self._embedder.embed_documents([c.chunk.text for c in chunks])

        candidates: list[tuple[int, int]] = []
        for (i, _), (j, _) in combinations(indexed, 2):
            if _cosine_similarity(vectors[i - 1], vectors[j - 1]) >= self._similarity_threshold:
                candidates.append((i, j))

        if not candidates:
            return ContradictionResult(has_contradiction=False)

        confirmed: list[ContradictionPair] = []
        for i, j in candidates:
            chunk_a, chunk_b = chunks[i - 1].chunk, chunks[j - 1].chunk
            verdict = self._judge_pair(chunk_a.text, chunk_b.text)
            if verdict is None:
                continue
            if verdict.get("contradiction") is True:
                confirmed.append(
                    ContradictionPair(
                        chunk_a_id=chunk_a.id,
                        chunk_b_id=chunk_b.id,
                        topic=str(verdict.get("topic", "")),
                        explanation=str(verdict.get("explanation", "")),
                        confidence=1.0,
                    )
                )

        if not confirmed:
            return ContradictionResult(has_contradiction=False)

        # Rebuild chunk_id -> [S#] lookup once, for the warning message.
        idx_by_chunk_id = {c.chunk.id: idx for idx, c in indexed}
        referenced_ids = {p.chunk_a_id for p in confirmed} | {p.chunk_b_id for p in confirmed}
        pair_refs = [f"[S{i}]" for i in sorted(idx_by_chunk_id[cid] for cid in referenced_ids)]
        warning_message = self._warning_template.format(pair_refs=", ".join(pair_refs))
        return ContradictionResult(
            has_contradiction=True,
            warning_message=warning_message,
            contradiction_pairs=confirmed,
        )

    def _judge_pair(self, text_a: str, text_b: str) -> dict[str, object] | None:
        """Ask the judge LLM whether ``text_a``/``text_b`` contradict. ``None`` on failure."""

        prompt = self._prompt_template.format(text_a=text_a, text_b=text_b)
        try:
            raw = self._judge_llm.generate(self._judge_system_prompt, prompt)
        except Exception as exc:
            logger.warning("contradiction judge LLM call failed: %s", exc)
            return None

        try:
            return _parse_judge_response(raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("contradiction judge returned unparsable output: %s", exc)
            return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors -- equals cosine similarity since embeddings from
    ``EmbeddingProvider`` are already L2-normalized project-wide (see rag/embeddings.py).
    """

    return sum(x * y for x, y in zip(a, b, strict=True))


def _parse_judge_response(raw: str) -> dict[str, object]:
    """Parse the judge's JSON, tolerating a leading/trailing markdown code fence."""

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    parsed: dict[str, object] = json.loads(text.strip())
    if "contradiction" not in parsed:
        raise KeyError("contradiction")
    return parsed
