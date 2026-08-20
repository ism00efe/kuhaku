"""Sparse (lexical) retrieval via a hand-rolled Okapi BM25.

Complements dense retrieval: embeddings capture meaning but blur exact tokens, while BM25
nails literal matches — error codes (``PAY-6006``), endpoint paths, and English technical
terms that appear verbatim inside otherwise-Turkish questions.

Implemented by hand (no ``rank_bm25``) to stay dependency-free and consistent with the
project's "minimal hand-rolled pipeline" decision (D4). The scoring is standard Okapi BM25:

    score(q, d) = Σ_t  IDF(t) · f(t,d)·(k1 + 1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
    IDF(t)      = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from kuhaku.tools.rag.models import Chunk, RetrievedChunk
from kuhaku.core.auth import AuthContext

from .chunking import Chunker, ParagraphChunker
from .config import RAGSettings
from .ingestion import load_corpus

logger = logging.getLogger("kuhaku.tools.rag.sparse_retriever")

# Unicode-aware word tokens, so Turkish diacritics (ı, ş, ğ, ü, ö, ç) survive.
# "PAY-5005" -> ["pay", "5005"]; the numeric part is highly discriminative.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` into word tokens."""

    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over an in-memory list of chunks.

    Satisfies the ``Retriever`` protocol in :mod:`.retriever`.
    """

    strategy = "sparse"

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b

        # Inverted index: term -> {doc_index: term_frequency}
        self._postings: dict[str, dict[int, int]] = {}
        self._doc_len: list[int] = []

        for idx, chunk in enumerate(self._chunks):
            tokens = tokenize(chunk.text)
            self._doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self._postings.setdefault(term, {})[idx] = freq

        n_docs = len(self._chunks)
        self._avgdl = (sum(self._doc_len) / n_docs) if n_docs else 0.0
        self._idf = {
            term: math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }
        logger.info("BM25 index built: %d chunks, %d terms", n_docs, len(self._postings))

    def count(self) -> int:
        return len(self._chunks)

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        auth_context: AuthContext | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` highest-scoring chunks (score > 0 only)."""

        if not self._chunks or top_k <= 0:
            return []

        scores: dict[int, float] = {}
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_idx, freq in postings.items():
                norm = 1 - self._b + self._b * (self._doc_len[doc_idx] / self._avgdl)
                contribution = idf * (freq * (self._k1 + 1)) / (freq + self._k1 * norm)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        # FR4: same freshness filter as DenseRetriever, so RRF fusion cannot resurface
        # an obsolete/out-of-window chunk via the sparse side (see DECISIONS.md D35).
        fresh = [
            RetrievedChunk(chunk=self._chunks[i], score=s)
            for i, s in ranked
            if self._chunks[i].is_fresh()
        ]

        # 4.1: same optional doc_type filter as DenseRetriever, same placement
        # (post-query, pre-fusion) so both retrieval paths filter identically.
        if doc_type is None:
            return fresh
        return [item for item in fresh if item.chunk.doc_type == doc_type]


def build_bm25_from_corpus(
    corpus_dir: str,
    *,
    chunk_size: int,
    overlap: int,
    k1: float = 1.5,
    b: float = 0.75,
    chunker: Chunker | None = None,
    rag_settings: RAGSettings | None = None,
) -> BM25Retriever:
    """Build a BM25 index from the corpus directory.

    Reuses :func:`load_corpus` rather than reading the vector store, which keeps the
    ``VectorStore`` abstraction unchanged. Importantly, ``load_corpus`` sanitizes at
    load, so the sparse index inherits the same PII guarantee as the dense index —
    reading the raw files here would have bypassed sanitization.

    ``chunker`` must match whatever chunker populated the dense (Chroma) index: RRF
    fusion (``retriever.reciprocal_rank_fusion``) keys purely by chunk id
    (``doc_id::index``), so a dense/sparse chunking-strategy mismatch would silently
    fuse the wrong chunk's text under a shared id, with no error anywhere. Defaults to
    ``ParagraphChunker`` (today's behavior) when omitted, same as ``ingest()``.

    ``rag_settings``, when given, supplies ``doc_type_prefix_mapping`` for doc-type
    inference (forwarded to :func:`load_corpus`), so sparse-indexed chunks get the same
    ``doc_type``s the dense index would.
    """

    chunker = chunker or ParagraphChunker()
    chunks: list[Chunk] = []
    for doc in load_corpus(corpus_dir, rag_settings=rag_settings):
        chunks.extend(chunker.chunk(doc, chunk_size=chunk_size, overlap=overlap))
    return BM25Retriever(chunks, k1=k1, b=b)
