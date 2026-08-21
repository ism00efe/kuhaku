"""Tests for domain models."""

from __future__ import annotations

from datetime import date

from kuhaku.tools.rag.models import Answer, Chunk, Citation, Document, RetrievedChunk


def test_chunk_metadata_shape():
    chunk = Chunk(
        id="d::0", document_id="d", title="Title", doc_type="faq",
        text="body", chunk_index=3, source_path="d.md",
    )
    meta = chunk.metadata()
    assert meta == {
        "document_id": "d",
        "title": "Title",
        "doc_type": "faq",
        "chunk_index": 3,
        "source_path": "d.md",
        "content_type": "text",
        "effective_date": "",
        "obsolete": False,
        "expiry_date": "",
        "access_tags": [],
    }


# --- FR4: freshness ------------------------------------------------------------
def _chunk(**overrides) -> Chunk:
    base = dict(
        id="d::0", document_id="d", title="Title", doc_type="faq",
        text="body", chunk_index=0, source_path="d.md",
    )
    base.update(overrides)
    return Chunk(**base)


def test_default_chunk_is_always_fresh():
    assert _chunk().is_fresh() is True


def test_obsolete_chunk_is_never_fresh():
    assert _chunk(obsolete=True).is_fresh() is False


def test_chunk_before_its_effective_date_is_not_fresh():
    chunk = _chunk(effective_date="2030-01-01")
    assert chunk.is_fresh(as_of=date(2025, 1, 1)) is False


def test_chunk_on_its_effective_date_is_fresh():
    chunk = _chunk(effective_date="2025-01-01")
    assert chunk.is_fresh(as_of=date(2025, 1, 1)) is True


def test_chunk_after_its_expiry_date_is_not_fresh():
    chunk = _chunk(expiry_date="2020-01-01")
    assert chunk.is_fresh(as_of=date(2025, 1, 1)) is False


def test_chunk_on_its_expiry_date_is_fresh():
    chunk = _chunk(expiry_date="2025-01-01")
    assert chunk.is_fresh(as_of=date(2025, 1, 1)) is True


def test_chunk_within_its_validity_window_is_fresh():
    chunk = _chunk(effective_date="2020-01-01", expiry_date="2030-01-01")
    assert chunk.is_fresh(as_of=date(2025, 1, 1)) is True


def test_obsolete_overrides_a_currently_valid_window():
    chunk = _chunk(effective_date="2020-01-01", expiry_date="2030-01-01", obsolete=True)
    assert chunk.is_fresh(as_of=date(2025, 1, 1)) is False


def test_document_defaults():
    doc = Document(id="d", title="T", doc_type="faq", text="x", source_path="d.md")
    assert doc.metadata == {}
    assert doc.effective_date == "" and doc.obsolete is False and doc.expiry_date == ""


def test_answer_composition():
    chunk = Chunk("d::0", "d", "T", "faq", "x", 0, "d.md")
    ans = Answer(
        text="hi [S1]",
        citations=[Citation("S1", "d", "T", "faq", "d.md", 0.9)],
        retrieved=[RetrievedChunk(chunk, 0.9)],
        redactions=["[EMAIL]×1"],
        trace_id="abc123",
    )
    assert ans.citations[0].document_id == "d"
    assert ans.retrieved[0].score == 0.9
    assert ans.redactions == ["[EMAIL]×1"]
    assert ans.trace_id == "abc123"


def test_answer_trace_id_defaults_to_empty_string():
    ans = Answer(text="hi", citations=[], retrieved=[])
    assert ans.trace_id == ""
