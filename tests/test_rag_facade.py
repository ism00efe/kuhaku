"""Unit tests for the top-level RAG facade (kuhaku.RAG): load_documents and the layered
system-prompt override surface (persona/language_policy/system_prompt)."""

import pytest

import kuhaku
from kuhaku import RAG
from kuhaku.core.auth import AuthContext
from kuhaku.core.config import Settings
from kuhaku.core.exceptions import SecurityComponentError
from kuhaku.core.security.guard import CANARY_TOKEN
from kuhaku.tools.rag.config import RAGSettings
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
    # cache=False: these tests don't exercise caching, and cache_enabled defaults to
    # True (Feature 2) -- without this, a test whose ask() reaches generation would
    # write a real ./data/kuhaku_qa_cache.sqlite3 relative to the test runner's cwd.
    # See tests/test_rag_cache_facade.py for the dedicated cache tests.
    rag = RAG(settings=settings, cache=False)
    return rag, fake_store, fake_embedder, fake_llm


@pytest.fixture
def build_rag(monkeypatch):
    """Factory fixture: build a fake-backed RAG() with arbitrary extra kwargs, for tests
    that need to pass e.g. ``persona=``/``language_policy=``/``system_prompt=``."""

    def _build(**kwargs):
        _patch_rag_deps(monkeypatch, FakeVectorStore(), FakeEmbeddings(), FakeLLM())
        settings = Settings(_env_file=None, audit_enabled=False)
        kwargs.setdefault("cache", False)  # see fake_rag_environment above
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

    def _fake_answer(question, context_text=None, *, auth_context=None):
        seen["question"] = question
        seen["context_text"] = context_text
        seen["auth_context"] = auth_context
        return kuhaku.Answer(text="ok", citations=[], retrieved=[])

    monkeypatch.setattr(rag.engine, "answer", _fake_answer)
    rag.ask("what happened?", "some structured blob")

    assert seen == {
        "question": "what happened?",
        "context_text": "some structured blob",
        "auth_context": None,
    }


def test_ask_forwards_auth_context_to_the_engine(monkeypatch, fake_rag_environment):
    rag, _store, _embedder, _llm = fake_rag_environment
    seen: dict[str, object] = {}

    def _fake_answer(question, context_text=None, *, auth_context=None):
        seen["auth_context"] = auth_context
        return kuhaku.Answer(text="ok", citations=[], retrieved=[])

    monkeypatch.setattr(rag.engine, "answer", _fake_answer)
    ctx = AuthContext(identity="u1", roles=("engineering",))
    rag.ask("what happened?", auth_context=ctx)

    assert seen["auth_context"] is ctx


def test_acceptance_document_level_access_filtering(fake_rag_environment):
    """The task's acceptance test, run through the public facade alone (no rag.engine
    access): a document tagged with access_tags is never retrieved, cited, or handed to
    the LLM for a caller whose roles don't include one of its tags, while an untagged
    document remains visible to everyone."""

    rag, _store, _embedder, llm = fake_rag_environment

    hr_text = "# Salary Policy\n\nSalary bands range from A to E based on level."
    eng_text = "# Deploy Runbook\n\nRun deploy.sh to ship a new release."

    rag.ingest(hr_text, filename="salary_policy.md", access_tags=["people_ops"])
    rag.ingest(eng_text, filename="deploy_runbook.md")  # no tags -> visible to all

    answer = rag.ask(
        "what are the salary bands",
        auth_context=AuthContext(identity="efe", roles=("engineering",)),
    )

    assert all("salary_policy" not in rc.chunk.source_path for rc in answer.retrieved)
    assert all("salary_policy" not in c.source_path for c in answer.citations)
    assert "Salary bands range" not in (llm.last_user or "")
    assert "people_ops" not in answer.text


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


# --- default retrieval strategy: hybrid, and re-ranker resolution --------------------


class _FakeCrossEncoderReranker:
    """Records the model name (and any retry kwargs) it was constructed with -- stands
    in for `CrossEncoderReranker` so these tests never reach `sentence_transformers`/
    HuggingFace, proving the setting turns re-ranking on without downloading anything."""

    instances: list[str] = []
    kwargs: list[dict] = []

    def __init__(self, model_name: str, **kwargs) -> None:
        _FakeCrossEncoderReranker.instances.append(model_name)
        _FakeCrossEncoderReranker.kwargs.append(kwargs)


class _ExplodingCrossEncoderReranker:
    """Fails the test if `RAG` ever constructs one -- the strongest possible proof that
    a bare `RAG()` builds no re-ranker (and therefore downloads no model)."""

    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("CrossEncoderReranker must not be constructed")


def test_bare_rag_uses_hybrid_retrieval_by_default(build_rag):
    rag = build_rag()
    assert rag.engine.get_retriever().strategy == "hybrid"


def test_explicit_retrieval_dense_still_means_dense(build_rag):
    rag = build_rag(retrieval="dense")
    assert rag.engine.get_retriever().strategy == "dense"


def test_explicit_invalid_retrieval_argument_raises(build_rag):
    with pytest.raises(ValueError, match="retrieval must be one of"):
        build_rag(retrieval="nonsense")


def test_invalid_retrieval_setting_from_env_var_raises(monkeypatch, build_rag):
    monkeypatch.setenv("KUHAKU_RAG__RETRIEVAL", "nonsense")
    with pytest.raises(ValueError, match="retrieval must be one of"):
        build_rag()


def test_bare_rag_constructs_no_reranker_and_downloads_nothing(monkeypatch, build_rag):
    monkeypatch.setattr(kuhaku, "CrossEncoderReranker", _ExplodingCrossEncoderReranker)
    rag = build_rag()
    assert "rerank" not in rag.engine.get_retriever().strategy


def test_rerank_enabled_env_var_turns_on_reranking_for_a_bare_rag(monkeypatch, build_rag):
    """The whole point of RAGSettings.rerank_enabled: turn re-ranking on from the
    environment without the caller passing `reranker=` at all."""

    _FakeCrossEncoderReranker.instances = []
    monkeypatch.setattr(kuhaku, "CrossEncoderReranker", _FakeCrossEncoderReranker)
    monkeypatch.setenv("KUHAKU_RAG__RERANK_ENABLED", "true")

    rag = build_rag()

    assert rag.engine.get_retriever().strategy == "hybrid+rerank"
    assert _FakeCrossEncoderReranker.instances == ["BAAI/bge-reranker-base"]


def test_explicit_reranker_false_beats_rerank_enabled_env_var(monkeypatch, build_rag):
    monkeypatch.setattr(kuhaku, "CrossEncoderReranker", _ExplodingCrossEncoderReranker)
    monkeypatch.setenv("KUHAKU_RAG__RERANK_ENABLED", "true")

    rag = build_rag(reranker=False)

    assert rag.engine.get_retriever().strategy == "hybrid"


# --- retry settings reach the vector store / re-ranker constructors (Feature 2) ------


class _SpyChromaVectorStore(FakeVectorStore):
    """Records the retry kwargs it was constructed with, on top of FakeVectorStore's
    normal in-memory behavior -- proves a RAGSettings retry-setting change reaches
    ChromaVectorStore's constructor without touching a real Chroma instance."""

    calls: list[dict] = []

    def __init__(self, persist_dir: str, collection_name: str, **kwargs) -> None:
        super().__init__()
        _SpyChromaVectorStore.calls.append(kwargs)


def test_vectorstore_retry_settings_reach_chromavectorstore_construction(monkeypatch):
    _SpyChromaVectorStore.calls = []
    monkeypatch.setattr(kuhaku, "ChromaVectorStore", _SpyChromaVectorStore)
    monkeypatch.setattr(kuhaku, "build_embedding_provider", lambda rs: FakeEmbeddings())
    monkeypatch.setattr(kuhaku, "build_llm_provider", lambda s: FakeLLM())

    rag_settings = RAGSettings(
        retry_enabled=False,
        retry_vectorstore_max_attempts=7,
        retry_vectorstore_backoff_seconds=2.5,
    )
    kuhaku.RAG(
        settings=Settings(_env_file=None, audit_enabled=False),
        rag_settings=rag_settings,
        cache=False,
    )

    assert _SpyChromaVectorStore.calls == [
        {
            "retry_enabled": False,
            "retry_max_attempts": 7,
            "retry_backoff_seconds": 2.5,
        }
    ]


def test_reranker_retry_settings_reach_crossencoderreranker_construction(monkeypatch, build_rag):
    """A re-ranker that is never constructed (the default) must not require its retry
    settings to be resolved -- this test only exercises the `rerank_enabled=True` path,
    where CrossEncoderReranker is actually built."""

    _FakeCrossEncoderReranker.instances = []
    _FakeCrossEncoderReranker.kwargs = []
    monkeypatch.setattr(kuhaku, "CrossEncoderReranker", _FakeCrossEncoderReranker)

    rag_settings = RAGSettings(
        rerank_enabled=True,
        retry_enabled=False,
        retry_reranker_max_attempts=9,
        retry_reranker_backoff_base_seconds=1.5,
        retry_reranker_backoff_max_seconds=8.0,
    )
    build_rag(rag_settings=rag_settings)

    assert _FakeCrossEncoderReranker.kwargs == [
        {
            "retry_enabled": False,
            "retry_max_attempts": 9,
            "retry_backoff_base_seconds": 1.5,
            "retry_backoff_max_seconds": 8.0,
        }
    ]


def test_guard_enabled_but_unwired_raises_at_construction(monkeypatch, tmp_path):
    """RAG() builds no GuardPipeline itself -- guard_enabled=True is something the
    caller explicitly opted into via Settings, so silently ignoring it would be exactly
    the "setting without a reader" mistake AGENTS.md warns against. This must fail at
    construction, not disappear silently."""

    _patch_rag_deps(monkeypatch, FakeVectorStore(), FakeEmbeddings(), FakeLLM())
    settings = Settings(
        _env_file=None,
        guard_enabled=True,
        audit_enabled=False,
    )

    with pytest.raises(SecurityComponentError, match="update_guard"):
        RAG(settings=settings, cache=False)


def test_guard_disabled_by_default_does_not_raise(monkeypatch):
    _patch_rag_deps(monkeypatch, FakeVectorStore(), FakeEmbeddings(), FakeLLM())
    settings = Settings(_env_file=None, audit_enabled=False)

    RAG(settings=settings, cache=False)  # must not raise


def test_unwritable_audit_log_path_warns_but_does_not_raise(monkeypatch, tmp_path, caplog):
    """Unlike guard_enabled, audit logging is on by default for everyone -- nobody
    opted into it explicitly through this construction call -- so a broken sink is
    reported (with the reason) and construction still succeeds, same as any other
    performance/helper component's fallback policy."""

    _patch_rag_deps(monkeypatch, FakeVectorStore(), FakeEmbeddings(), FakeLLM())
    unwritable_parent = tmp_path / "not_a_directory"
    unwritable_parent.write_text("x", encoding="utf-8")  # a file, not a directory
    settings = Settings(
        _env_file=None,
        audit_enabled=True,
        audit_log_path=str(unwritable_parent / "audit.jsonl"),
    )

    with caplog.at_level("WARNING"):
        RAG(settings=settings, cache=False)  # must not raise

    assert "audit log" in caplog.text


# --- generic kwargs -> Settings/RAGSettings forwarding --------------------------------


def test_unknown_settings_field_is_forwarded_without_a_dedicated_kwarg(build_rag):
    """A Settings field with no dedicated RAG() parameter (llm_timeout_seconds) is still
    settable directly -- the generic kwargs -> Settings/RAGSettings forwarding closes
    the gap without RAG.__init__ needing to grow a new named parameter for it."""

    rag = build_rag(llm_timeout_seconds=7)
    assert rag._settings.llm_timeout_seconds == 7


def test_unknown_ragsettings_field_is_forwarded_without_a_dedicated_kwarg(build_rag):
    rag = build_rag(chunk_size=999)
    assert rag._rag_settings.chunk_size == 999


def test_field_shared_by_both_settings_classes_is_forwarded_via_settings_and_reaches_ragsettings(
    build_rag,
):
    """retry_enabled exists on both Settings and RAGSettings (the latter mirrors the
    former via RAGSettings.from_settings()) -- forwarding it once, to Settings, is
    enough to reach both, since RAGSettings is derived from the overridden Settings."""

    rag = build_rag(retry_enabled=False)
    assert rag._settings.retry_enabled is False
    assert rag._rag_settings.retry_enabled is False


def test_truly_unknown_kwarg_still_raises_type_error(monkeypatch):
    _patch_rag_deps(monkeypatch, FakeVectorStore(), FakeEmbeddings(), FakeLLM())

    with pytest.raises(TypeError, match="unexpected keyword argument 'not_a_real_setting'"):
        RAG(not_a_real_setting=True)
