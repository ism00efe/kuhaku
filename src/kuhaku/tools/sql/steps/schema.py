"""Protected step: work out what this caller may see, and what the model is shown."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from kuhaku.core.exceptions import ConfigError

from ..access import EntitledSchema, EntitledTable, entitled_schema
from ..context import SQLContext
from ..pipeline import SCHEMA_STEP, Step
from ..result import Issue, IssueKind

Pruner = Callable[[str, Sequence[EntitledTable]], Sequence[EntitledTable]]
"""``(question, entitled tables) -> the tables to show the model``. A pruner only ever
receives the entitled tables, so it cannot widen what the model sees; names it returns
that are not among them are ignored."""

_WORD = re.compile(r"[a-z0-9]+")

NO_MATCH_MESSAGE = "the question cannot be answered from the available data"


class SchemaStep(Step):
    """Builds the caller's :class:`EntitledSchema` and the part of it the model sees.

    With ``max_tables`` set (and no ``pruner``), the tables are ranked by how many words
    of the question appear in their names, column names and descriptions, and the top
    ``max_tables`` are shown -- cheap relevance pruning for a wide schema. Pruning only
    narrows the prompt; the validator still accepts any entitled table, so a model that
    names a pruned-away table it is entitled to is not rejected.

    An empty entitled schema ends the question as ``NO_MATCH`` without calling the
    model -- the same outcome as a question nothing in the database answers.
    """

    name = SCHEMA_STEP

    def __init__(self, *, max_tables: int | None = None, pruner: Pruner | None = None) -> None:
        if max_tables is not None and max_tables < 1:
            raise ConfigError("max_tables must be at least 1.")
        self.max_tables = max_tables
        self.pruner = pruner

    def run(self, ctx: SQLContext) -> None:
        schema = entitled_schema(ctx.catalog, ctx.policy, ctx.auth_context)
        ctx.schema = schema
        if not len(schema):
            ctx.issues.append(Issue(IssueKind.NO_MATCH, "no_match", NO_MATCH_MESSAGE))
            ctx.stop()
            return
        ctx.prompt_schema = self._prune(ctx.question, schema)
        ctx.metadata["prompt_tables"] = list(ctx.prompt_schema.names)

    def _prune(self, question: str, schema: EntitledSchema) -> EntitledSchema:
        tables = schema.tables
        if self.pruner is not None:
            chosen = self.pruner(question, tables)
            return schema.subset(entry.name for entry in chosen) if chosen else schema
        if self.max_tables is None or len(tables) <= self.max_tables:
            return schema
        words = _words(question)
        ranked = sorted(
            enumerate(tables), key=lambda pair: (-_score(pair[1], words), pair[0])
        )
        return schema.subset(entry.name for _, entry in ranked[: self.max_tables])


def _words(text: str) -> set[str]:
    words = set()
    for word in _WORD.findall(text.lower()):
        words.add(word)
        if len(word) > 3 and word.endswith("s"):
            words.add(word[:-1])
    return words


def _score(entry: EntitledTable, question_words: set[str]) -> int:
    table = entry.table
    text = " ".join(
        [table.name, table.description]
        + [f"{column.name} {column.description}" for column in table.columns]
    ).replace("_", " ")
    return len(question_words & _words(text))
