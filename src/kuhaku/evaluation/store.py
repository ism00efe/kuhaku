"""Persistence for per-sample evaluation results: the ``EvaluationStore`` protocol
plus an in-memory and a SQLite-backed implementation.

Deliberately independent of ``EvaluationRunner`` -- anything that can accumulate
``(sample_id, result_ids, metrics, answer)`` rows per ``run_id`` and average them back
out satisfies the protocol, runner-agnostic. Both ``sample_id`` and ``result_ids`` are
plain strings/ids -- no RAG type (or any other tool's type) appears in this module.

Writing a ``(run_id, sample_id)`` pair that already exists *replaces* the earlier row
instead of appending a second one: a re-run under the same id is a retry, and two rows
for one sample made ``summary()`` average both.

``ReadableEvaluationStore`` is the wider contract for reading a run's rows back. It is a separate protocol rather than two more
methods on ``EvaluationStore`` so a third-party store written against the original
protocol keeps satisfying it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EvaluationStore(Protocol):
    """Persists per-sample evaluation results and aggregates them."""

    def add_result(
        self,
        sample_id: str,
        result_ids: list[str],
        metrics: dict[str, float],
        answer: str | None = None,
        *,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def summary(self, run_id: str | None = None) -> dict[str, float]: ...


@runtime_checkable
class ReadableEvaluationStore(EvaluationStore, Protocol):
    """An ``EvaluationStore`` whose rows can be read back.

    Every row is a dict with ``question_id``, ``retrieved_ids``, ``metrics``, ``answer``,
    ``run_id`` and ``metadata`` -- the shape ``InMemoryEvaluationStore.results`` has
    always exposed.
    """

    def get_results(self, run_id: str | None = None) -> list[dict[str, Any]]: ...

    def list_runs(self) -> list[str]: ...


class InMemoryEvaluationStore:
    """Default ``EvaluationStore``: keeps results in a list, averages per run on ``summary()``."""

    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []
        self._run_ids: list[str] = []

    def add_result(
        self,
        sample_id: str,
        result_ids: list[str],
        metrics: dict[str, float],
        answer: str | None = None,
        *,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if run_id not in self._run_ids:
            self._run_ids.append(run_id)
        row = {
            "question_id": sample_id,
            "retrieved_ids": result_ids,
            "metrics": metrics,
            "answer": answer,
            "run_id": run_id,
            "metadata": metadata or {},
        }
        for index, existing in enumerate(self._results):
            if existing["run_id"] == run_id and existing["question_id"] == sample_id:
                self._results[index] = row
                return
        self._results.append(row)

    def summary(self, run_id: str | None = None) -> dict[str, float]:
        if run_id is None:
            if not self._run_ids:
                return {}
            run_id = self._run_ids[-1]
        return _average_metrics(r["metrics"] for r in self._results if r["run_id"] == run_id)

    def get_results(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Rows of ``run_id`` (the latest run when omitted), in insertion order."""

        if run_id is None:
            if not self._run_ids:
                return []
            run_id = self._run_ids[-1]
        return [r for r in self._results if r["run_id"] == run_id]

    def list_runs(self) -> list[str]:
        """Run ids in the order each run first wrote a row."""

        return list(self._run_ids)

    @property
    def results(self) -> list[dict[str, Any]]:
        return list(self._results)


class SqliteEvaluationStore:
    """``EvaluationStore`` backed by a SQLite table (``eval_runs``) at ``db_path``."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                retrieved_ids TEXT NOT NULL,
                metrics TEXT NOT NULL,
                answer TEXT,
                metadata TEXT
            )
            """
        )
        # A database written before metadata was persisted lacks the column; adding it in
        # place keeps its existing runs readable (their rows read back with ``{}``).
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(eval_runs)")}
        if "metadata" not in columns:
            self._conn.execute("ALTER TABLE eval_runs ADD COLUMN metadata TEXT")
        self._conn.commit()

    def add_result(
        self,
        sample_id: str,
        result_ids: list[str],
        metrics: dict[str, float],
        answer: str | None = None,
        *,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Replace, not append -- see the module docstring. The new row takes a fresh id,
        # so it also becomes the run's latest write.
        self._conn.execute(
            "DELETE FROM eval_runs WHERE run_id = ? AND question_id = ?", (run_id, sample_id)
        )
        self._conn.execute(
            "INSERT INTO eval_runs "
            "(run_id, question_id, retrieved_ids, metrics, answer, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                sample_id,
                json.dumps(result_ids),
                json.dumps(metrics),
                answer,
                # ensure_ascii=False keeps non-ASCII expected/actual values readable when
                # the database is inspected by hand; default=str stops one exotic value
                # from losing the whole row.
                json.dumps(metadata or {}, ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()

    def _latest_run_id(self) -> str | None:
        row = self._conn.execute(
            "SELECT run_id FROM eval_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else row[0]

    def summary(self, run_id: str | None = None) -> dict[str, float]:
        if run_id is None:
            run_id = self._latest_run_id()
            if run_id is None:
                return {}

        rows = self._conn.execute(
            "SELECT metrics FROM eval_runs WHERE run_id = ?", (run_id,)
        ).fetchall()
        return _average_metrics(json.loads(row[0]) for row in rows)

    def get_results(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Rows of ``run_id`` (the latest run when omitted), in insertion order."""

        if run_id is None:
            run_id = self._latest_run_id()
            if run_id is None:
                return []
        rows = self._conn.execute(
            "SELECT question_id, retrieved_ids, metrics, answer, run_id, metadata "
            "FROM eval_runs WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [
            {
                "question_id": question_id,
                "retrieved_ids": json.loads(retrieved_ids),
                "metrics": json.loads(metrics),
                "answer": answer,
                "run_id": row_run_id,
                "metadata": json.loads(metadata) if metadata else {},
            }
            for question_id, retrieved_ids, metrics, answer, row_run_id, metadata in rows
        ]

    def list_runs(self) -> list[str]:
        """Run ids in the order each run first wrote a row."""

        rows = self._conn.execute(
            "SELECT run_id FROM eval_runs GROUP BY run_id ORDER BY MIN(id)"
        ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self._conn.close()


def _average_metrics(all_metrics: Any) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in all_metrics:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}
