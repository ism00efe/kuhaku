"""Golden evaluation dataset loading and validation.

Independent of any application's corpus or RAG pipeline -- reads the kuhaku-wide
JSONL golden dataset schema (``question_id``, ``question``, ``expected_sources``,
``golden_chunks``, ``golden_answer``, ``category``) and nothing else.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("kuhaku.evaluation.dataset")

_REQUIRED_FIELDS = ("question_id", "question")
_OPTIONAL_FIELDS = ("expected_sources", "golden_chunks", "golden_answer", "category")


@dataclass(frozen=True)
class EvaluationItem:
    """One labeled question in a golden evaluation dataset -- tool-agnostic: nothing here
    is specific to RAG or any other tool built on kuhaku."""

    question_id: str
    question: str
    expected_sources: list[str]
    golden_chunks: list[str]
    golden_answer: str
    category: str | None


def _require_str(value: object, field: str, line_no: int) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"line {line_no}: field {field!r} must be a string, got {type(value).__name__}"
        )
    return value


def _require_str_list(value: object, field: str, line_no: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"line {line_no}: field {field!r} must be a list of strings")
    return value


def _parse_item(raw: dict, line_no: int) -> EvaluationItem:
    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"line {line_no}: missing required field(s): {', '.join(missing)}")

    for field in _OPTIONAL_FIELDS:
        if field not in raw:
            logger.warning("line %d: missing optional field %r", line_no, field)

    question_id = _require_str(raw["question_id"], "question_id", line_no)
    question = _require_str(raw["question"], "question", line_no)

    expected_sources = _require_str_list(raw.get("expected_sources", []), "expected_sources", line_no)
    golden_chunks = _require_str_list(raw.get("golden_chunks", []), "golden_chunks", line_no)

    golden_answer = raw.get("golden_answer", "")
    golden_answer = _require_str(golden_answer, "golden_answer", line_no)

    category = raw.get("category")
    if category is not None:
        category = _require_str(category, "category", line_no)

    return EvaluationItem(
        question_id=question_id,
        question=question,
        expected_sources=expected_sources,
        golden_chunks=golden_chunks,
        golden_answer=golden_answer,
        category=category,
    )


def load_eval_dataset(path: str | Path) -> list[EvaluationItem]:
    """Load and validate a golden evaluation dataset from a JSONL file.

    Raises ``ValueError`` (with the offending line number) for malformed JSON, a
    missing required field, or a field of the wrong type. Missing optional fields are
    logged as warnings and filled with an empty default (``[]``, ``""``, or ``None``).
    """

    path = Path(path)
    items: list[EvaluationItem] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: malformed JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"line {line_no}: expected a JSON object, got {type(raw).__name__}")
        items.append(_parse_item(raw, line_no))
    return items
