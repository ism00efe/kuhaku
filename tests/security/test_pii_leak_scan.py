"""Dedicated PII-leak regression suite.

Generates synthetic content with REAL (but fake) PII in every category
``sanitization.py`` claims to mask, runs it through the actual ingestion path, the
actual sparse index, and a full ``RAGEngine.answer()`` call — capturing everything the
pipeline logs along the way — and asserts the raw values never appear anywhere: not in
indexed chunk text, not in the prompt sent to the LLM, not in the LLM's response
(simulated by a stub that echoes its input, the worst case if a model repeats context
verbatim), and not in any captured log line.

This is the shape of test that would have caught the BM25 sparse-index PII issue found
during development (the index was originally built by reading raw corpus files instead
of the already-sanitized ``load_corpus`` output) — and is meant to catch the next one,
in whatever new code path introduces it.
"""

from __future__ import annotations

import logging

import pytest

from kuhaku.tools.rag.models import Chunk
from kuhaku.core.sanitization import sanitize_text
from kuhaku.tools.rag.engine import RAGEngine
from kuhaku.tools.rag.ingestion import chunk_document, ingest, load_corpus
from kuhaku.tools.rag.retriever import DenseRetriever
from kuhaku.tools.rag.sparse_retriever import build_bm25_from_store
from tests.conftest import FakeEmbeddings, FakeVectorStore

# Fake-but-format-valid PII covering every category sanitization.py claims to mask.
RAW_CARD = "4111 1111 1111 1111"  # Luhn-valid test PAN
RAW_TCKN = "10000000146"  # checksum-valid test TCKN
RAW_EMAIL = "musteri.destek@ornekbanka.com"
RAW_IP = "203.0.113.77"
RAW_PHONE = "+90 532 111 22 33"
RAW_TOKEN = "Bearer sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

ALL_RAW_PII = [RAW_CARD, RAW_TCKN, RAW_EMAIL, RAW_IP, RAW_PHONE, RAW_TOKEN]


class EchoingLLM:
    """Worst-case LLM stub: echoes the exact prompt it was given back as the answer.

    Simulates a model that repeats its context verbatim — the failure mode that
    surfaces a sanitization gap most directly, since anything present in the prompt
    shows up unchanged in the "response" too.
    """

    name = "echo"

    def __init__(self) -> None:
        self.last_system: str | None = None
        self.last_user: str | None = None

    def generate(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return user


def _assert_no_raw_pii(haystack: str, *, where: str) -> None:
    for pii in ALL_RAW_PII:
        assert pii not in haystack, f"raw PII '{pii}' leaked into {where}"


def _build_engine(chunks: list[Chunk]) -> tuple[RAGEngine, EchoingLLM]:
    embedder = FakeEmbeddings()
    store = FakeVectorStore(chunks)
    llm = EchoingLLM()
    engine = RAGEngine(embedder, store, llm, top_k=4, retriever=DenseRetriever(embedder, store))
    return engine, llm


# --- 1) Ingestion path: raw PII must never survive load_corpus ----------------
def test_ingestion_strips_all_pii_categories(tmp_path):
    (tmp_path / "log_leak_test.json").write_text(
        f'{{"error_code": "PAY-1001", "customer": {{"email": "{RAW_EMAIL}", '
        f'"phone": "{RAW_PHONE}", "national_id": "{RAW_TCKN}"}}, '
        f'"card": {{"pan": "{RAW_CARD}"}}, "client_ip": "{RAW_IP}", '
        f'"auth": "{RAW_TOKEN}"}}',
        encoding="utf-8",
    )

    docs = load_corpus(str(tmp_path))
    assert len(docs) == 1
    _assert_no_raw_pii(docs[0].text, where="ingested document text")

    for chunk in chunk_document(docs[0], chunk_size=500, overlap=50):
        _assert_no_raw_pii(chunk.text, where="chunk text")


# --- 2) Sparse index: must inherit sanitization (the historical regression) ---
def test_sparse_index_never_contains_raw_pii(tmp_path):
    """The sparse index is now built from the vector store's own chunks (see
    sparse_retriever.build_bm25_from_store), not by re-reading corpus_dir -- so this
    drives the real ingest() -> store -> BM25 path end to end, the same chunks the
    dense side would search, to prove the store-sourced index inherits ingest()'s
    sanitization rather than bypassing it."""

    (tmp_path / "log_leak_test.xml").write_text(
        f"<log><email>{RAW_EMAIL}</email><ip>{RAW_IP}</ip>"
        f"<message>card {RAW_CARD} phone {RAW_PHONE} id {RAW_TCKN}</message></log>",
        encoding="utf-8",
    )

    store = FakeVectorStore()
    ingest(str(tmp_path), FakeEmbeddings(), store, chunk_size=500, overlap=50)

    bm25 = build_bm25_from_store(store)
    indexed_text = "\n".join(c.text for c in store.iter_chunks())
    _assert_no_raw_pii(indexed_text, where="BM25 sparse index")
    # The masks themselves should be present — proving the text was sanitized, not
    # silently dropped.
    assert "[EMAIL]" in indexed_text and "[CARD]" in indexed_text
    # And the raw values must not be searchable through the retriever itself either.
    assert bm25.retrieve(RAW_EMAIL, top_k=5) == []
    assert bm25.retrieve(RAW_CARD, top_k=5) == []


# --- 3) Full query path: prompt + "response" + retrieved chunks --------------
def test_full_pipeline_query_containing_pii_never_leaks(tmp_path):
    """A user who pastes raw PII directly into their QUESTION must never see it reach
    the retriever, the LLM prompt, or the LLM's (echoed) response."""

    (tmp_path / "faq_test.md").write_text(
        "# FAQ\n\nQ: How do refunds work?\nA: Refunds take 1-5 business days.\n",
        encoding="utf-8",
    )
    chunks: list[Chunk] = []
    for doc in load_corpus(str(tmp_path)):
        chunks.extend(chunk_document(doc, chunk_size=500, overlap=50))
    engine, llm = _build_engine(chunks)

    dirty_question = (
        f"Kartım {RAW_CARD} ile bir sorun var, TCKN {RAW_TCKN}, "
        f"e-posta {RAW_EMAIL}, IP {RAW_IP}, telefon {RAW_PHONE}, token {RAW_TOKEN}. "
        "Refund ne kadar sürer?"
    )

    answer = engine.answer(dirty_question)

    _assert_no_raw_pii(answer.text, where="Answer.text")
    _assert_no_raw_pii(llm.last_user or "", where="LLM prompt (build_user_prompt output)")
    _assert_no_raw_pii(llm.last_system or "", where="LLM system prompt")
    for item in answer.retrieved:
        _assert_no_raw_pii(item.chunk.text, where="retrieved chunk text")

    # And the redaction report must show every category was actually caught.
    joined = " ".join(answer.redactions)
    for tag in ("[CARD]", "[NATIONAL_ID]", "[EMAIL]", "[IP]", "[PHONE]", "[TOKEN]"):
        assert tag in joined, f"expected {tag} in redaction report"


def test_full_pipeline_uploaded_context_containing_pii_never_leaks():
    """Same as above but PII arrives via the uploaded-context path, not the question."""

    engine, llm = _build_engine([])  # empty KB is fine; we only inspect sanitization

    dirty_context = (
        f'{{"error_code": "PAY-1001", "email": "{RAW_EMAIL}", "pan": "{RAW_CARD}", '
        f'"national_id": "{RAW_TCKN}", "ip": "{RAW_IP}", "phone": "{RAW_PHONE}", '
        f'"token": "{RAW_TOKEN}"}}'
    )
    answer = engine.answer("Bu log neden declined oldu?", context_text=dirty_context)

    # KB is empty, so the engine short-circuits before generation — but sanitization
    # and redaction reporting must have already run on the raw log before that point.
    joined = " ".join(answer.redactions)
    for tag in ("[CARD]", "[NATIONAL_ID]", "[EMAIL]", "[IP]", "[PHONE]", "[TOKEN]"):
        assert tag in joined, f"expected {tag} in redaction report"
    _assert_no_raw_pii(answer.text, where="Answer.text")


# --- 4) Log files: nothing captured by the logging system may contain raw PII -
def test_no_raw_pii_in_captured_logs(tmp_path, caplog):
    """The most important regression guard: even with structured logging and
    step-by-step instrumentation wired through the engine, no log line anywhere may
    contain a raw PII value. Catches a careless `logger.info(query)` anywhere in the
    pipeline, present or future."""

    (tmp_path / "faq_test.md").write_text("# FAQ\n\nQ: test\nA: test\n", encoding="utf-8")
    chunks: list[Chunk] = []
    for doc in load_corpus(str(tmp_path)):
        chunks.extend(chunk_document(doc, chunk_size=500, overlap=50))
    engine, _ = _build_engine(chunks)

    dirty_question = (
        f"Kart {RAW_CARD}, TCKN {RAW_TCKN}, mail {RAW_EMAIL}, ip {RAW_IP}, "
        f"tel {RAW_PHONE}, token {RAW_TOKEN} ile ilgili bir sorun var."
    )
    dirty_context = f'{{"pan": "{RAW_CARD}", "email": "{RAW_EMAIL}"}}'

    with caplog.at_level(logging.DEBUG):
        engine.answer(dirty_question, context_text=dirty_context)

    _assert_no_raw_pii(caplog.text, where="captured log output")


# --- 5) Sanitization function itself, exhaustively, once more in this suite ---
@pytest.mark.parametrize("raw", ALL_RAW_PII)
def test_sanitize_text_masks_every_pii_category(raw):
    clean, redactions = sanitize_text(f"context around {raw} more context")
    assert raw not in clean
    assert redactions  # at least one category was recorded
