"""Tests for chunking strategies (ParagraphChunker, StructuralChunker)."""

from __future__ import annotations

import json

import pytest

from kuhaku.tools.rag.chunking import (
    ParagraphChunker,
    StructuralChunker,
    build_chunker,
    chunk_document,
)
from kuhaku.tools.rag.config import RAGSettings
from kuhaku.tools.rag.models import Document


def _doc(text: str, *, doc_id: str = "d", doc_type: str = "faq", **freshness) -> Document:
    return Document(
        id=doc_id, title="T", doc_type=doc_type, text=text, source_path=f"{doc_id}.md", **freshness
    )


# --- ParagraphChunker ---------------------------------------------------------

def test_paragraph_chunker_matches_chunk_document():
    doc = _doc("first paragraph\n\nsecond paragraph")
    assert ParagraphChunker().chunk(doc, chunk_size=1000, overlap=50) == chunk_document(
        doc, 1000, 50
    )


def test_paragraph_chunker_default_content_type_is_text():
    chunks = ParagraphChunker().chunk(_doc("body text"), chunk_size=1000, overlap=50)
    assert all(c.content_type == "text" for c in chunks)


# --- StructuralChunker: heading splitting --------------------------------------

def test_structural_chunker_splits_by_heading():
    doc = _doc("# A\n\nintro to A\n\n## A.1\n\ncontent of A.1\n\n# B\n\ncontent of B")
    texts = [
        c.text for c in StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
        if c.content_type == "text"
    ]
    assert any("intro to A" in t and not t.startswith("A >") for t in texts)
    assert any(t.startswith("A > A.1") and "content of A.1" in t for t in texts)
    assert any(t.startswith("B") and "content of B" in t for t in texts)


def test_structural_chunker_no_headings_falls_back_to_single_section():
    doc = _doc("just prose\n\nmore prose, no headings anywhere")
    chunks = StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].content_type == "text"
    assert "just prose" in chunks[0].text and "more prose" in chunks[0].text


def test_structural_chunker_empty_text_returns_empty_list():
    assert StructuralChunker().chunk(_doc("   "), chunk_size=1000, overlap=50) == []


def test_structural_chunker_respects_chunk_size_for_oversized_section():
    long_para = "kelime " * 100
    doc = _doc(f"# Başlık\n\n{long_para}")
    chunks = StructuralChunker().chunk(doc, chunk_size=120, overlap=20)
    text_chunks = [c for c in chunks if c.content_type == "text"]
    assert len(text_chunks) > 1


# --- StructuralChunker: table extraction ---------------------------------------

_TABLE_MD = (
    "| Kod | Anlam |\n"
    "|---|---|\n"
    "| RC-05 | Do Not Honor |\n"
    "| RC-51 | Insufficient Funds |"
)


def test_structural_chunker_extracts_table_as_json():
    doc = _doc(f"# Hata Kodları\n\n{_TABLE_MD}")
    chunks = StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
    table_chunks = [c for c in chunks if c.content_type == "table"]
    assert len(table_chunks) == 1
    assert json.loads(table_chunks[0].text) == {
        "columns": ["Kod", "Anlam"],
        "rows": [
            {"Kod": "RC-05", "Anlam": "Do Not Honor"},
            {"Kod": "RC-51", "Anlam": "Insufficient Funds"},
        ],
    }


def test_structural_chunker_table_inside_a_section_leaves_surrounding_prose_intact():
    doc = _doc(f"# Hata Kodları\n\nGiriş cümlesi.\n\n{_TABLE_MD}\n\nSon cümle.")
    chunks = StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
    assert sum(1 for c in chunks if c.content_type == "table") == 1
    text_chunks = [c for c in chunks if c.content_type == "text"]
    assert any("Giriş cümlesi" in c.text for c in text_chunks)
    assert any("Son cümle" in c.text for c in text_chunks)
    assert not any("|" in c.text for c in text_chunks)


def test_prose_with_a_stray_pipe_is_not_misread_as_a_table():
    doc = _doc("# Örnek\n\nBu satırda cmd1 | cmd2 gibi bir işaret var ama bu bir tablo değildir.")
    chunks = StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
    assert not any(c.content_type == "table" for c in chunks)


# --- StructuralChunker: glossary extraction ------------------------------------

def test_structural_chunker_extracts_glossary_entries_bold_form():
    doc = _doc("# Tanımlar\n\n**provizyon**: tutarın hesapta bloke edilmesi.")
    glossary = [
        c for c in StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
        if c.content_type == "glossary"
    ]
    assert len(glossary) == 1
    assert json.loads(glossary[0].text) == [
        {"term": "provizyon", "definition": "tutarın hesapta bloke edilmesi."}
    ]


def test_structural_chunker_extracts_glossary_entries_plain_form():
    doc = _doc("# Tanımlar\n\nTerim: Açıklama metni.")
    glossary = [
        c for c in StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
        if c.content_type == "glossary"
    ]
    assert len(glossary) == 1
    assert json.loads(glossary[0].text) == [{"term": "Terim", "definition": "Açıklama metni."}]


def test_structural_chunker_extracts_bulleted_glossary_entry():
    doc = _doc("# Tanımlar\n\n- **ters ibraz**: chargeback süreci.")
    glossary = [
        c for c in StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
        if c.content_type == "glossary"
    ]
    assert len(glossary) == 1
    assert json.loads(glossary[0].text)[0]["term"] == "ters ibraz"


@pytest.mark.parametrize(
    "line",
    [
        "14:30: Connection timeout",
        "https://example.com/path: bir açıklama",
        "Not: bu alan zorunludur, boş bırakılamaz, kontrol ediniz baştan sona.",
        "Örnek: kullanım şekli aşağıdaki gibidir.",
    ],
)
def test_structural_chunker_glossary_false_positive_guards(line):
    doc = _doc(f"# Bölüm\n\n{line}")
    chunks = StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
    assert not any(c.content_type == "glossary" for c in chunks)


# --- build_chunker factory ------------------------------------------------------

def test_build_chunker_selects_paragraph_by_default():
    settings = RAGSettings()
    assert isinstance(build_chunker(settings), ParagraphChunker)


def test_build_chunker_selects_structural():
    settings = RAGSettings(chunking_strategy="structural")
    assert isinstance(build_chunker(settings), StructuralChunker)


def test_build_chunker_rejects_unknown_strategy():
    settings = RAGSettings(chunking_strategy="bogus")
    with pytest.raises(ValueError, match="bogus"):
        build_chunker(settings)


# --- FR4: freshness metadata inherited from the parent Document ----------------
def test_paragraph_chunker_threads_freshness_fields_onto_every_chunk():
    doc = _doc(
        "first paragraph\n\nsecond paragraph",
        obsolete=True, effective_date="2025-01-01", expiry_date="2027-01-01",
    )
    chunks = ParagraphChunker().chunk(doc, chunk_size=1000, overlap=50)
    assert len(chunks) >= 1
    assert all(c.obsolete is True for c in chunks)
    assert all(c.effective_date == "2025-01-01" for c in chunks)
    assert all(c.expiry_date == "2027-01-01" for c in chunks)


def test_structural_chunker_threads_freshness_fields_onto_every_chunk():
    doc = _doc("# A\n\nintro\n\n# B\n\nbody", obsolete=True, expiry_date="2020-01-01")
    chunks = StructuralChunker().chunk(doc, chunk_size=1000, overlap=50)
    assert len(chunks) >= 1
    assert all(c.obsolete is True for c in chunks)
    assert all(c.expiry_date == "2020-01-01" for c in chunks)
