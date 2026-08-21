"""Unit tests for the top-level RAG facade (kuhaku.RAG): load_documents and the layered
system-prompt override surface (persona/language_policy/system_prompt)."""

import pytest

import kuhaku
from kuhaku import RAG
from kuhaku.core.config import Settings
from kuhaku.core.security.guard import CANARY_TOKEN
from kuhaku.tools.rag.ingestion import UnsupportedFileType
from kuhaku.tools.rag.messages import DEFAULT_ENGINE_MESSAGES
from kuhaku.tools.rag.prompts import SYSTEM_PROMPT
from tests.conftest import FakeEmbeddings, FakeLLM, FakeVectorStore


def _patch_rag_deps(monkeypatch, fake_store, fake_embedder, fake_llm):
    monkeypatch.setattr(kuhaku, "build_embedding_provider", lambda rs: fake_embedder)
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", lambda *a, **k: fake_store)
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s: fake_llm)


@pytest.fixture
def fake_rag_environment(monkeypatch):
    """Patch RAG internals to use in-memory fakes without any external network or Chroma calls."""
    fake_store = FakeVectorStore()
    fake_embedder = FakeEmbeddings()
    fake_llm = FakeLLM()

    _patch_rag_deps(monkeypatch, fake_store, fake_embedder, fake_llm)

    settings = Settings(_env_file=None, audit_enabled=False)
    rag = RAG(settings=settings)
    return rag, fake_store, fake_embedder, fake_llm


@pytest.fixture
def build_rag(monkeypatch):
    """Factory fixture: build a fake-backed RAG() with arbitrary extra kwargs, for tests
    that need to pass e.g. ``persona=``/``language_policy=``/``system_prompt=``."""

    def _build(**kwargs):
        _patch_rag_deps(monkeypatch, FakeVectorStore(), FakeEmbeddings(), FakeLLM())
        settings = Settings(_env_file=None, audit_enabled=False)
        return RAG(settings=settings, **kwargs)

    return _build


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


# --- system prompt override surface (persona / language_policy / system_prompt) ------


def test_rag_default_system_prompt_is_the_framework_default(fake_rag_environment):
    rag, _store, _embedder, _llm = fake_rag_environment
    assert rag.engine.get_system_prompt() == SYSTEM_PROMPT


def test_rag_persona_override_replaces_only_the_persona_layer(build_rag):
    custom_persona = "You are a friendly museum tour-guide chatbot."
    rag = build_rag(persona=custom_persona)
    prompt = rag.engine.get_system_prompt()

    assert custom_persona in prompt
    # the safety core -- canary rule and mandatory citations -- survives untouched
    assert CANARY_TOKEN in prompt
    assert "Citations (mandatory)" in prompt
    assert "Instruction precedence" in prompt


def test_rag_language_policy_override_replaces_only_that_layer(build_rag):
    custom_policy = "## Output language\nAlways answer in German, regardless of input."
    rag = build_rag(language_policy=custom_policy)
    prompt = rag.engine.get_system_prompt()

    assert custom_policy in prompt
    assert CANARY_TOKEN in prompt
    assert "Citations (mandatory)" in prompt


def test_rag_system_prompt_override_replaces_the_whole_thing(build_rag):
    full_custom = "Answer questions about penguins only."
    rag = build_rag(system_prompt=full_custom)

    assert rag.engine.get_system_prompt() == full_custom


def test_rag_system_prompt_override_takes_precedence_over_persona(build_rag):
    """`system_prompt` replaces the prompt outright -- a `persona` passed alongside it
    is ignored rather than silently layered in."""

    full_custom = "Answer questions about penguins only."
    rag = build_rag(system_prompt=full_custom, persona="This should be ignored.")

    assert rag.engine.get_system_prompt() == full_custom


# --- ask()/chat_repl() ---------------------------------------------------------------


def test_ask_forwards_context_text_to_the_engine(monkeypatch, fake_rag_environment):
    rag, _store, _embedder, _llm = fake_rag_environment
    seen: dict[str, object] = {}

    def _fake_answer(question, context_text=None):
        seen["question"] = question
        seen["context_text"] = context_text
        return kuhaku.Answer(text="ok", citations=[], retrieved=[])

    monkeypatch.setattr(rag.engine, "answer", _fake_answer)
    rag.ask("what happened?", "some structured blob")

    assert seen == {"question": "what happened?", "context_text": "some structured blob"}


def test_chat_repl_prints_the_engines_own_abstention_text(
    monkeypatch, fake_rag_environment, capsys
):
    """chat_repl must display whatever text the engine attached to an abstained Answer,
    not a hard-coded literal of its own -- see prompts.ABSTENTION_PHRASE, exported
    precisely so nothing downstream carries a copied literal that could drift."""
    rag, _store, _embedder, _llm = fake_rag_environment

    abstained_answer = kuhaku.Answer(
        text="I don't have enough information to answer this question.",
        citations=[],
        retrieved=[],
        abstained=True,
    )
    monkeypatch.setattr(rag, "ask", lambda question: abstained_answer)

    inputs = iter(["what is the meaning of life?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    rag.chat_repl()

    captured = capsys.readouterr()
    assert abstained_answer.text in captured.out
    assert "Üzgünüm" not in captured.out
    assert "Sources:" not in captured.out


# --- sparse / hybrid retrieval sourced from the store, not corpus_dir (Feature 2) ----


def test_hybrid_construction_and_query_with_no_corpus_dir_configured(build_rag):
    """RAG(retrieval="hybrid") must not require corpus_dir at all -- BM25 is built from
    whatever is already in the vector store."""

    rag = build_rag(retrieval="hybrid")
    answer = rag.ask("anything")
    assert answer.text == DEFAULT_ENGINE_MESSAGES.empty_kb


def test_ingest_after_construction_appears_in_hybrid_results(build_rag):
    rag = build_rag(retrieval="hybrid")
    rag.ingest("Refunds take 1-5 business days for PAY-9911 disputed cases.", "faq.md")

    results = rag.engine.retrieve("PAY-9911", top_k=5)
    assert any("PAY-9911" in r.chunk.text for r in results)


def test_ingest_after_construction_appears_in_sparse_results(build_rag):
    rag = build_rag(retrieval="sparse")
    rag.ingest("Refunds take 1-5 business days for PAY-9911 disputed cases.", "faq.md")

    results = rag.engine.retrieve("PAY-9911", top_k=5)
    assert any("PAY-9911" in r.chunk.text for r in results)


def test_repeated_ingests_all_become_searchable_via_sparse(build_rag):
    rag = build_rag(retrieval="sparse")
    for i in range(4):
        rag.ingest(f"Document number {i} discusses topic PAY-{9000 + i}.", f"doc_{i}.md")

    for i in range(4):
        results = rag.engine.retrieve(f"PAY-{9000 + i}", top_k=5)
        assert any(f"PAY-{9000 + i}" in r.chunk.text for r in results)


def test_hybrid_retrieval_on_empty_store_returns_no_results_without_crashing(build_rag):
    rag = build_rag(retrieval="hybrid")
    assert rag.engine.retrieve("anything", top_k=5) == []


def test_sparse_retrieval_on_empty_store_returns_no_results_without_crashing(build_rag):
    rag = build_rag(retrieval="sparse")
    assert rag.engine.retrieve("anything", top_k=5) == []


def test_sparse_engine_answer_on_empty_store_distinguishes_empty_kb_from_no_chunks(build_rag):
    """An empty store ("nothing has been ingested") must not read the same as a query
    that found no matching chunks -- RAGEngine._answer() already gates on store.count()
    before ever calling retrieve(), independent of retrieval strategy, so this behavior
    was already correct for dense and carries over unchanged for sparse/hybrid."""

    rag = build_rag(retrieval="sparse")
    answer = rag.ask("anything")
    assert answer.text == DEFAULT_ENGINE_MESSAGES.empty_kb
    assert answer.text != DEFAULT_ENGINE_MESSAGES.no_chunks
