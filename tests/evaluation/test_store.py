from __future__ import annotations

import sqlite3

from kuhaku.evaluation.store import (
    InMemoryEvaluationStore,
    ReadableEvaluationStore,
    SqliteEvaluationStore,
)


def test_in_memory_store_summary_averages_across_results():
    store = InMemoryEvaluationStore()
    store.add_result("q1", ["a"], {"recall": 1.0, "mrr": 1.0}, run_id="run1")
    store.add_result("q2", ["b"], {"recall": 0.0, "mrr": 0.0}, run_id="run1")

    assert store.summary() == {"recall": 0.5, "mrr": 0.5}


def test_in_memory_store_empty_summary_is_empty_dict():
    assert InMemoryEvaluationStore().summary() == {}


def test_in_memory_store_summary_defaults_to_latest_run():
    store = InMemoryEvaluationStore()
    store.add_result("q1", ["a"], {"recall": 0.0}, run_id="run1")
    store.add_result("q2", ["b"], {"recall": 1.0}, run_id="run2")

    assert store.summary() == {"recall": 1.0}
    assert store.summary(run_id="run1") == {"recall": 0.0}


def test_in_memory_store_summary_unknown_run_id_is_empty_dict():
    store = InMemoryEvaluationStore()
    store.add_result("q1", ["a"], {"recall": 1.0}, run_id="run1")

    assert store.summary(run_id="does-not-exist") == {}


def test_in_memory_store_records_generic_sample_id_and_result_ids():
    store = InMemoryEvaluationStore()
    store.add_result("sample-42", ["doc-a", "doc-b"], {"recall": 1.0}, run_id="run1")

    assert store.results[0]["question_id"] == "sample-42"
    assert store.results[0]["retrieved_ids"] == ["doc-a", "doc-b"]


def test_in_memory_store_accepts_optional_metadata():
    store = InMemoryEvaluationStore()
    store.add_result("q1", ["a"], {"recall": 1.0}, run_id="run1", metadata={"category": "faq"})

    assert store.results[0]["metadata"] == {"category": "faq"}


def test_in_memory_store_metadata_defaults_to_empty_dict():
    store = InMemoryEvaluationStore()
    store.add_result("q1", ["a"], {"recall": 1.0}, run_id="run1")

    assert store.results[0]["metadata"] == {}


def test_sqlite_store_round_trip(tmp_path):
    db_path = tmp_path / "eval.db"
    store = SqliteEvaluationStore(db_path)

    store.add_result("q1", ["a", "b"], {"recall": 1.0, "mrr": 0.5}, answer="the answer", run_id="run1")
    store.add_result("q2", ["c"], {"recall": 0.0, "mrr": 0.0}, run_id="run1")

    summary = store.summary()
    store.close()

    assert summary == {"recall": 0.5, "mrr": 0.25}
    assert db_path.exists()


def test_sqlite_store_persists_across_reopen(tmp_path):
    db_path = tmp_path / "eval.db"
    store1 = SqliteEvaluationStore(db_path)
    store1.add_result("q1", ["a"], {"recall": 1.0}, run_id="run1")
    store1.close()

    store2 = SqliteEvaluationStore(db_path)
    assert store2.summary() == {"recall": 1.0}
    store2.close()


def test_sqlite_store_summary_defaults_to_latest_run(tmp_path):
    db_path = tmp_path / "eval.db"
    store = SqliteEvaluationStore(db_path)

    store.add_result("q1", ["a"], {"recall": 0.0}, run_id="run1")
    store.add_result("q2", ["b"], {"recall": 1.0}, run_id="run2")

    assert store.summary() == {"recall": 1.0}
    assert store.summary(run_id="run1") == {"recall": 0.0}
    store.close()


def test_sqlite_store_summary_unknown_run_id_is_empty_dict(tmp_path):
    db_path = tmp_path / "eval.db"
    store = SqliteEvaluationStore(db_path)
    store.add_result("q1", ["a"], {"recall": 1.0}, run_id="run1")

    assert store.summary(run_id="does-not-exist") == {}
    store.close()


def test_sqlite_store_accepts_optional_metadata_without_error(tmp_path):
    db_path = tmp_path / "eval.db"
    store = SqliteEvaluationStore(db_path)
    store.add_result("q1", ["a"], {"recall": 1.0}, run_id="run1", metadata={"category": "faq"})

    assert store.summary() == {"recall": 1.0}
    store.close()


def test_sqlite_store_persists_metadata(tmp_path):
    store = SqliteEvaluationStore(tmp_path / "eval.db")
    store.add_result("q1", ["a"], {"recall": 1.0}, run_id="run1", metadata={"category": "çay"})

    assert store.get_results("run1")[0]["metadata"] == {"category": "çay"}
    store.close()


def test_sqlite_store_get_results_round_trips_every_column(tmp_path):
    db_path = tmp_path / "eval.db"
    store = SqliteEvaluationStore(db_path)
    store.add_result("q1", ["a", "b"], {"recall": 1.0}, answer="yes", run_id="run1")
    store.add_result("q2", [], {"recall": 0.0}, run_id="run2")
    store.close()

    reopened = SqliteEvaluationStore(db_path)
    assert reopened.get_results("run1") == [
        {
            "question_id": "q1",
            "retrieved_ids": ["a", "b"],
            "metrics": {"recall": 1.0},
            "answer": "yes",
            "run_id": "run1",
            "metadata": {},
        }
    ]
    assert [r["question_id"] for r in reopened.get_results()] == ["q2"]
    assert reopened.get_results("nope") == []
    reopened.close()


def test_sqlite_store_lists_runs_in_first_write_order(tmp_path):
    store = SqliteEvaluationStore(tmp_path / "eval.db")
    assert store.list_runs() == []
    store.add_result("q1", [], {}, run_id="b")
    store.add_result("q1", [], {}, run_id="a")
    store.add_result("q2", [], {}, run_id="b")

    assert store.list_runs() == ["b", "a"]
    store.close()


def test_sqlite_store_migrates_a_database_without_the_metadata_column(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE eval_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, "
        "question_id TEXT NOT NULL, retrieved_ids TEXT NOT NULL, metrics TEXT NOT NULL, "
        "answer TEXT)"
    )
    conn.execute(
        "INSERT INTO eval_runs (run_id, question_id, retrieved_ids, metrics, answer) "
        "VALUES ('old', 'q1', '[]', '{\"recall\": 1.0}', NULL)"
    )
    conn.commit()
    conn.close()

    store = SqliteEvaluationStore(db_path)
    store.add_result("q2", [], {"recall": 0.0}, run_id="new", metadata={"k": 1})

    assert store.get_results("old")[0]["metadata"] == {}
    assert store.get_results("new")[0]["metadata"] == {"k": 1}
    assert store.summary("old") == {"recall": 1.0}
    store.close()


def test_in_memory_store_get_results_and_list_runs():
    store = InMemoryEvaluationStore()
    assert store.get_results() == []
    store.add_result("q1", [], {"recall": 1.0}, run_id="run1")
    store.add_result("q2", [], {"recall": 0.0}, run_id="run2")

    assert store.list_runs() == ["run1", "run2"]
    assert [r["question_id"] for r in store.get_results()] == ["q2"]
    assert [r["question_id"] for r in store.get_results("run1")] == ["q1"]


def test_both_stores_satisfy_the_readable_protocol(tmp_path):
    sqlite_store = SqliteEvaluationStore(tmp_path / "eval.db")
    assert isinstance(InMemoryEvaluationStore(), ReadableEvaluationStore)
    assert isinstance(sqlite_store, ReadableEvaluationStore)
    sqlite_store.close()


def test_same_run_and_sample_replaces_the_row_in_memory():
    store = InMemoryEvaluationStore()
    store.add_result("q1", ["a"], {"recall": 0.0}, run_id="run1")
    store.add_result("q1", ["a"], {"recall": 1.0}, run_id="run1")

    assert len(store.get_results("run1")) == 1
    assert store.summary("run1") == {"recall": 1.0}


def test_same_run_and_sample_replaces_the_row_in_sqlite(tmp_path):
    store = SqliteEvaluationStore(tmp_path / "eval.db")
    store.add_result("q1", ["a"], {"recall": 0.0}, run_id="run1")
    store.add_result("q2", ["a"], {"recall": 0.0}, run_id="run1")
    store.add_result("q1", ["a"], {"recall": 1.0}, run_id="run1")

    rows = store.get_results("run1")
    assert [r["question_id"] for r in rows] == ["q2", "q1"]
    assert store.summary("run1") == {"recall": 0.5}
    store.close()
