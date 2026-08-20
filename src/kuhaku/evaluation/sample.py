"""``EvaluationSample``: the tool-agnostic unit every metric scores.

Any tool that can be reduced to "a query went in, this text/these context strings/these
tool calls came out" produces one of these -- ``EvaluationSample`` carries no RAG-specific
type (no ``RetrievedChunk``, no ``Chunk``), so a non-RAG target (a plain classifier, a
code-search tool, ...) is just as scoreable as ``RAGEngine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class EvaluationSample:
    """One evaluated interaction: a query, what the target produced, and (once merged
    with a golden dataset item by ``EvaluationRunner``) what it was expected to produce.

    ``contexts`` holds the textual context the target used to answer (e.g. retrieved
    chunk text) -- consumed by generation/faithfulness metrics. ``retrieved_doc_ids`` is
    the ordered list of identifiers the target actually returned (e.g. retrieved chunk
    document ids); ``relevant_doc_ids`` is the golden set they are graded against.
    Retrieval metrics compare these two id collections, never ``contexts`` text, since a
    tool-agnostic target has no guarantee its context strings double as stable ids.

    ``tool_calls`` is a free-form list for targets that invoke sub-tools (function/tool
    calling agents) -- no metric in this package reads it yet, but it is part of the
    tool-agnostic contract so a future tool-use metric has somewhere to look.

    ``metadata`` carries anything else a metric or the runner needs that doesn't warrant
    its own field -- e.g. ``golden_answer`` (the reference text ``AnswerCorrectnessMetric``
    compares against) and ``question_id``, both set by ``EvaluationRunner`` when it merges
    a target's sample with its golden dataset item.
    """

    query: str
    answer: str | None = None
    contexts: list[str] | None = None
    relevant_doc_ids: set[str] | None = None
    retrieved_doc_ids: list[str] | None = None
    tool_calls: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_golden(
        self, *, relevant_doc_ids: set[str], metadata: dict[str, Any]
    ) -> "EvaluationSample":
        """Return a copy with golden-dataset fields merged in.

        Never overwrites a ``relevant_doc_ids`` a target already set itself -- targets
        aren't expected to know golden labels, but nothing stops one from producing its
        own relevance judgment. Metadata is merged with the golden values taking
        precedence, since it comes from the trusted dataset rather than the target.
        """

        merged_relevant = self.relevant_doc_ids if self.relevant_doc_ids is not None else relevant_doc_ids
        merged_metadata = {**self.metadata, **metadata}
        return replace(self, relevant_doc_ids=merged_relevant, metadata=merged_metadata)
