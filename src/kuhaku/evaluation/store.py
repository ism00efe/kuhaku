"""Persistence for per-sample evaluation results: the ``EvaluationStore`` protocol
plus an in-memory and a SQLite-backed implementation.

Deliberately independent of ``EvaluationRunner`` -- anything that can accumulate
``(sample_id, result_ids, metrics, answer)`` rows per ``run_id`` and average them back
out satisfies the protocol, runner-agnostic. Both ``sample_id`` and ``result_ids`` are
plain strings/ids -- no RAG type (or any other tool's type) appears in this module.
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
        self._results.append(
            {
                "question_id": sample_id,
                "retrieved_ids": result_ids,
                "metrics": metrics,
                "answer": answer,
                "run_id": run_id,
                "metadata": metadata or {},
            }
        )

    def summary(self, run_id: str | None = None) -> dict[str, float]:
        if run_id is None:
            if not self._run_ids:
                return {}
            run_id = self._run_ids[-1]
        return _average_metrics(r["metrics"] for r in self._results if r["run_id"] == run_id)

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
                answer TEXT
            )
            """
        )
        self._conn.commit()

    def add_result(
        self,
        sample_id: str,
        result_ids: list[str],
        metrics: dict[str, float],
        answer: str | None = None,
        *,
        run_id: str,
        metadata: dict[str, Any] | None = None,  # noqa: ARG002 -- not yet persisted to SQLite
    ) -> None:
        self._conn.execute(
            "INSERT INTO eval_runs (run_id, question_id, retrieved_ids, metrics, answer) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, sample_id, json.dumps(result_ids), json.dumps(metrics), answer),
        )
        self._conn.commit()

    def summary(self, run_id: str | None = None) -> dict[str, float]:
        if run_id is None:
            row = self._conn.execute(
                "SELECT run_id FROM eval_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return {}
            run_id = row[0]

        rows = self._conn.execute(
            "SELECT metrics FROM eval_runs WHERE run_id = ?", (run_id,)
        ).fetchall()
        return _average_metrics(json.loads(row[0]) for row in rows)

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
