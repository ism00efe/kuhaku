"""Unit tests for the top-level RAG facade (kuhaku.RAG), focusing on load_documents."""

import pytest

import kuhaku
from kuhaku import RAG
from kuhaku.core.config import Settings
from tests.conftest import FakeEmbeddings, FakeLLM, FakeVectorStore
from kuhaku.tools.rag.ingestion import UnsupportedFileType


@pytest.fixture
def fake_rag_environment(monkeypatch):
    """Patch RAG internals to use in-memory fakes without any external network or Chroma calls."""
    fake_store = FakeVectorStore()
    fake_embedder = FakeEmbeddings()
    fake_llm = FakeLLM()

    monkeypatch.setattr(kuhaku, "build_embedding_provider", lambda rs: fake_embedder)
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: fake_store)
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s: fake_llm)

    settings = Settings(_env_file=None, audit_enabled=False)
    rag = RAG(settings=settings)
    return rag, fake_store, fake_embedder, fake_llm


def test_load_documents_indexes_supported_files(tmp_path, fake_rag_environment):
    rag, store, embedder, _llm = fake_rag_environment

    doc1 = tmp_path / "guide.md"
    doc1.write_text("# API Guide\nThis is a sample guide content.", encoding="utf-8")

    doc2 = tmp_path / "faq.txt"
    doc2.write_text("Q: How does this work?\nA: It works via RAG.", encoding="utf-8")

    # Unsupported file should be ignored
    doc3 = tmp_path / "photo.png"
    doc3.write_bytes(b"\x89PNG\r\n\x1a\n")

    count = rag.load_documents(tmp_path)

    assert count == 2
    assert store.count() >= 2
    assert len(embedder.doc_calls) >= 2


def test_load_documents_with_pdf_extraction(tmp_path, fake_rag_environment, monkeypatch):
    rag, store, embedder, _llm = fake_rag_environment

    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy pdf bytes")

    monkeypatch.setattr(
        kuhaku,
        "extract_text",
        lambda filename, data: "Extracted content from PDF document",
    )

    count = rag.load_documents(tmp_path)
    assert count == 1
    assert store.count() >= 1


def test_load_documents_nonexistent_directory_raises(fake_rag_environment, tmp_path):
    rag, _store, _embedder, _llm = fake_rag_environment
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="no such directory"):
        rag.load_documents(missing_dir)


def test_load_documents_empty_directory_returns_zero(tmp_path, fake_rag_environment):
    rag, store, _embedder, _llm = fake_rag_environment
    empty_dir = tmp_path / "empty_folder"
    empty_dir.mkdir()

    count = rag.load_documents(empty_dir)
    assert count == 0
    assert store.count() == 0


def test_load_documents_skips_unreadable_or_corrupted_file(
    tmp_path, fake_rag_environment, monkeypatch
):
    rag, store, _embedder, _llm = fake_rag_environment

    valid_doc = tmp_path / "valid.txt"
    valid_doc.write_text("Valid text file content.", encoding="utf-8")

    bad_pdf = tmp_path / "corrupted.pdf"
    bad_pdf.write_bytes(b"corrupted bytes")

    def _raise_unsupported(filename, data):
        raise UnsupportedFileType("PDF parsing failed")

    monkeypatch.setattr(kuhaku, "extract_text", _raise_unsupported)

    count = rag.load_documents(tmp_path)
    # Only valid.txt was indexed; bad_pdf was skipped gracefully
    assert count == 1
    assert store.count() >= 1
