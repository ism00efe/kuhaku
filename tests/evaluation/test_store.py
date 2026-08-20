from __future__ import annotations

from kuhaku.evaluation.store import InMemoryEvaluationStore, SqliteEvaluationStore


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
